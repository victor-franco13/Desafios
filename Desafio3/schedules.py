"""Agendas de taxa de aprendizado para o Desafio 3.

build_schedule(name, optimizer, total_steps, warmup_steps=0) devolve um
torch.optim.lr_scheduler._LRScheduler (ou objeto com .step() compatível),
chamado uma vez por passo de otimização (sched.step() logo após opt.step()).
"""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def build_schedule(name: str, optimizer: torch.optim.Optimizer,
                    total_steps: int, warmup_steps: int = 0,
                    base_lr: float = None):
    name = name.lower()

    if name == "constante":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    if name == "cosseno":
        def fn(step):
            progress = min(step / max(1, total_steps), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda=fn)

    if name == "warmup_cosseno":
        def fn(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(progress, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda=fn)

    if name == "one_cycle":
        if base_lr is None:
            base_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps,
                           pct_start=0.1)

    raise ValueError(f"Agenda desconhecida: {name}")


SCHEDULES = ["constante", "cosseno", "warmup_cosseno", "one_cycle"]
