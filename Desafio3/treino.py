#!/usr/bin/env python3
"""
Desafio 3 - GBC073: treino.py configurável (otimizador, eta, agenda, semente)

Problema: DQN em ALE/Pong-v5 com CNN (projeto do grupo: RL em jogos Atari).

Uso:
    python treino.py --optimizer adamw --lr 3e-4 --schedule cosseno \
        --seed 0 --steps 200000 --tag baseline

Gera:
    logs/<tag>_<otimizador>_<agenda>_lr<lr>_seed<seed>.csv   (instrumentação)
    experimentos.csv                                          (uma linha, append)
"""
import argparse
import csv
import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics,
)

from muon import Muon
from schedules import build_schedule, SCHEDULES

OPTIMIZERS = ["sgd", "sgd_nesterov", "adam", "adamw", "muon"]

# ----------------------------------------------------------------------------
# Ambiente
# ----------------------------------------------------------------------------

def make_env(env_id: str, seed: int):
    env = gym.make(env_id, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84,
                              grayscale_obs=True, scale_obs=False)
    env = FrameStackObservation(env, 4)
    env = RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ----------------------------------------------------------------------------
# Rede (Nature CNN)
# ----------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, n_actions)  # camada de saída: fica no AdamW

    def forward(self, x):
        # x: (B, 4, 84, 84) uint8 -> float [0,1]
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.head(x)


# ----------------------------------------------------------------------------
# Replay buffer simples
# ----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_obs[idx]).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )


# ----------------------------------------------------------------------------
# Otimizadores / grupos de parâmetros
# ----------------------------------------------------------------------------

def build_optimizer(name: str, net: nn.Module, lr: float):
    """Devolve (lista_de_optimizers, lista_de_nomes_de_grupo).
    Para Muon: dois otimizadores (Muon nas camadas ocultas 2D/4D; AdamW no
    resto - vieses e a camada 'head'). Para os demais: um único otimizador
    sobre todos os parâmetros."""
    name = name.lower()

    if name == "muon":
        muon_params = [p for n, p in net.named_parameters()
                        if p.ndim >= 2 and "head" not in n]
        rest_params = [p for n, p in net.named_parameters()
                        if not (p.ndim >= 2 and "head" not in n)]
        opts = [
            Muon(muon_params, lr=lr, momentum=0.95, weight_decay=0.01),
            torch.optim.AdamW(rest_params, lr=3e-4, weight_decay=0.01),
        ]
        return opts

    params = net.parameters()
    if name == "sgd":
        return [torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False)]
    if name == "sgd_nesterov":
        return [torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)]
    if name == "adam":
        return [torch.optim.Adam(params, lr=lr)]
    if name == "adamw":
        return [torch.optim.AdamW(params, lr=lr, weight_decay=0.01)]
    raise ValueError(f"Otimizador desconhecido: {name}")


# ----------------------------------------------------------------------------
# Instrumentação
# ----------------------------------------------------------------------------

LAYER_NAMES = ["conv1", "fc1", "head"]  # primeira, uma do meio, última


def global_grad_norm(net) -> float:
    total = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def global_param_norm(net) -> float:
    total = 0.0
    for p in net.parameters():
        total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def snapshot_layers(net):
    return {name: dict(net.named_parameters())[f"{name}.weight"].detach().clone()
            for name in LAYER_NAMES}


def update_ratios(net, before: dict):
    out = {}
    params = dict(net.named_parameters())
    for name, old in before.items():
        new = params[f"{name}.weight"].detach()
        denom = old.norm().item() + 1e-12
        out[name] = (new - old).norm().item() / denom
    return out


# ----------------------------------------------------------------------------
# Loop de treino
# ----------------------------------------------------------------------------

def train(args):
    os.makedirs("logs", exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env, args.seed)
    n_actions = env.action_space.n

    net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizers = build_optimizer(args.optimizer, net, args.lr)
    warmup_steps = max(1, int(0.05 * args.steps))
    schedulers = [build_schedule(args.schedule, opt, args.steps, warmup_steps,
                                  base_lr=args.lr) for opt in optimizers]

    buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)

    run_name = f"{args.tag}_{args.optimizer}_{args.schedule}_lr{args.lr:g}_seed{args.seed}"
    log_path = os.path.join("logs", f"{run_name}.csv")
    log_rows = []

    eps_start, eps_end, eps_decay_steps = 1.0, 0.05, int(0.1 * args.steps)

    obs, _ = env.reset(seed=args.seed)
    episode_returns = deque(maxlen=20)
    diverged = False
    t0 = time.time()

    step = 0
    while step < args.steps:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if random.random() < eps or len(buffer) < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(device)
                action = net(obs_t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs

        if "episode" in info:
            episode_returns.append(info["episode"]["r"])

        if done:
            obs, _ = env.reset()

        # --- passo de otimização ---
        if len(buffer) >= args.learning_starts:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(args.batch_size)

            with torch.no_grad():
                next_q = target_net(next_obs_b).max(dim=1).values
                target = rew_b + args.gamma * (1 - done_b) * next_q

            q = net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q, target)

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()

            grad_norm = global_grad_norm(net)
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

            before = snapshot_layers(net) if step % args.log_every == 0 else None

            for opt in optimizers:
                opt.step()
            for sch in schedulers:
                sch.step()

            if step % args.target_update_every == 0:
                target_net.load_state_dict(net.state_dict())

            if not torch.isfinite(loss) or grad_norm > 1e4:
                diverged = True

            if step % args.log_every == 0:
                param_norm = global_param_norm(net)
                ratios = update_ratios(net, before) if before else {}
                lr_now = optimizers[0].param_groups[0]["lr"]
                val_metric = float(np.mean(episode_returns)) if episode_returns else float("nan")
                row = dict(
                    passo=step, perda=loss.item(), grad_norm=grad_norm,
                    param_norm=param_norm, lr=lr_now, retorno_medio=val_metric,
                    diverged=int(diverged),
                )
                for name in LAYER_NAMES:
                    row[f"rho_{name}"] = ratios.get(name, float("nan"))
                log_rows.append(row)

            if diverged and args.stop_on_divergence:
                break

        step += 1

    elapsed = time.time() - t0

    # salva log detalhado
    if log_rows:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    # append no experimentos.csv
    final_metric = float(np.mean(episode_returns)) if episode_returns else float("nan")
    exp_row = dict(
        tag=args.tag, otimizador=args.optimizer, agenda=args.schedule,
        lr=args.lr, seed=args.seed, passos=step, metrica_validacao=final_metric,
        divergiu=int(diverged), tempo_s=round(elapsed, 1), log=log_path,
    )
    exp_path = "experimentos.csv"
    write_header = not os.path.exists(exp_path)
    with open(exp_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(exp_row)

    print(f"[{run_name}] passos={step} metrica={final_metric:.2f} "
          f"divergiu={diverged} tempo={elapsed:.0f}s -> {log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ALE/Pong-v5")
    p.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
    p.add_argument("--schedule", choices=SCHEDULES, default="constante")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200_000,
                    help="Orçamento de passos de treino (reduzido e igual "
                         "para todas as comparações, conforme regras do "
                         "Desafio 3).")
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target_update_every", type=int, default=1_000)
    p.add_argument("--clip_grad_norm", type=float, default=10.0)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--tag", default="run")
    p.add_argument("--stop_on_divergence", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
