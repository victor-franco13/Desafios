"""
desafio2_CaioCouto_VictorFranco.py — Desafio 2 (GBC073): ativação e inicialização em redes profundas

Ideia: o desvio dos pesos é CALCULADO para a ativação escolhida, não decorado.
Numa camada z' = W f(z) + b com W ~ (0, s²) e z ~ N(0, 1):

    Var(z')          = fan_in · s² · E[f(z)²]      (ida: sinal)
    Var(grad entra)  = fan_out · s² · E[f'(z)²]    (volta: gradiente, por unidade)

Critério "variancia": fan_in · s² · E[f²]  = 1  -> a variância fica em 1 nas L camadas.
Critério "gradiente": fan_in · s² · E[f'²] = 1  -> o gradiente não some nem explode.
Para ReLU os dois coincidem (E[f²] = E[f'²] = 1/2) e dão He. Para tanh o segundo dá
s² = 1/fan_in (tanh na criticalidade); o primeiro deixaria tanh no regime caótico.

Detalhes:
  * camada 1: a entrada já vem padronizada (E[x²] ≈ 1), então s² = 1/fan_in, sem E[f²];
  * pesos ortogonais reescalados ao desvio calculado (menos flutuação entre camadas);
  * camada de logits com desvio reduzido: a rede começa prevendo ~uniforme e o SGD
    com momento 0,9 não dá passos enormes no início;
  * f(x) = ganho · g(x): o E[f²] absorve o ganho, então a rede no passo 0 é a mesma,
    mas a taxa de aprendizado efetiva das camadas 2..L muda por ganho².

Só torch e a biblioteca padrão. E[f²] e E[f'²] são estimados uma vez por Monte Carlo.
"""
import math
from functools import partial

import torch
import torch.nn.functional as F

# Configuração escolhida (ver experimentos_d2.py para a comparação entre candidatos):
# LeakyReLU(0,2) + pesos ortogonais + ganho 0,5 teve S = 114,0 no modo completo e foi a mais
# estável em L = 48 (MNIST 0,951 / Fashion 0,810 / CIFAR-10 0,384; ReLU + He fica na chance).
CONFIG = dict(
    ativ="lrelu",             # relu, lrelu, gelu, silu, elu, selu, tanh
    alfa=0.2,                 # inclinação negativa da lrelu (ignorado pelas demais)
    ganho=0.5,                # f(x) = ganho * g(x): taxa efetiva das camadas ocultas × 0,25
    criterio="variancia",     # "variancia" ou "gradiente"
    pesos="ortogonal",        # "normal" ou "ortogonal"
    escala_saida=0.1,         # multiplica o desvio da última camada
)


def _base(nome: str, alfa: float):
    return {
        "relu": torch.relu,
        "lrelu": partial(F.leaky_relu, negative_slope=alfa),
        "gelu": F.gelu,
        "silu": F.silu,
        "elu": F.elu,
        "selu": F.selu,
        "tanh": torch.tanh,
    }[nome]


_cache_momentos: dict = {}


def _momentos(g) -> tuple[float, float]:
    """E[g(z)²] e E[g'(z)²] com z ~ N(0,1), por Monte Carlo (semente fixa)."""
    if g not in _cache_momentos:
        gen = torch.Generator().manual_seed(0)
        z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
        with torch.enable_grad():
            z.requires_grad_(True)
            y = g(z)
            (dy,) = torch.autograd.grad(y.sum(), z)
        _cache_momentos[g] = (y.detach().pow(2).mean().item(), dy.pow(2).mean().item())
    return _cache_momentos[g]


def _fazer(cfg: dict):
    """Devolve (ativacao, inicializar) para uma configuração."""
    g = _base(cfg["ativ"], cfg.get("alfa", 0.2))
    a = float(cfg.get("ganho", 1.0))
    E_g2, E_dg2 = _momentos(g)
    m = a * a * (E_g2 if cfg.get("criterio", "variancia") == "variancia" else E_dg2)

    def ativacao(x: torch.Tensor) -> torch.Tensor:
        return a * g(x) if a != 1.0 else g(x)

    @torch.no_grad()
    def inicializar(W: torch.Tensor, b: torch.Tensor,
                    fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
        desvio = math.sqrt(1.0 / (fan_in * (1.0 if camada == 1 else m)))
        if camada == n_camadas:
            desvio *= cfg.get("escala_saida", 1.0)
        if desvio == 0.0:
            W.zero_()
        elif cfg.get("pesos", "normal") == "ortogonal":
            torch.nn.init.orthogonal_(W)
            W.mul_(desvio / W.pow(2).mean().sqrt())
        else:
            W.normal_(0.0, desvio)
        b.zero_()

    return ativacao, inicializar


ativacao, inicializar = _fazer(CONFIG)
