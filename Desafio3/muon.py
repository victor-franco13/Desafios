"""
Implementação própria do otimizador Muon (Jordan, 2024), para matrizes 2-D
(e kernels de convolução achatados em (C_out, C_in*k_h*k_w)).

Para cada parâmetro:
  1. acumula momento M (SGD com momento, como de costume)
  2. em vez de dar o passo com M, ortogonaliza M via Newton-Schulz (5 iterações)
     e dá o passo com essa versão ortogonalizada

Isso faz cada direção singular da matriz de atualização receber
aproximadamente o mesmo tamanho de passo — inclusive as direções raras que
SGD/Adam tendem a ignorar.

Uso: ver treino.py — parâmetros 2-D "ocultos" (não cabeça, não embedding) vão
para este otimizador; vieses, normalizações e a camada de saída vão para AdamW.
"""
import torch
from torch.optim import Optimizer


def newton_schulz_ortho(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Ortogonalização aproximada de G via iteração de Newton-Schulz.
    G: (m, n). Retorna uma matriz de mesma forma, com valores singulares ~1.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(Optimizer):
    """Muon para parâmetros 2-D. Parâmetros com ndim==4 (kernels de
    convolução) devem ser passados já achatados pelo chamador (ver
    treino.py: flatten_conv / unflatten_conv), ou o chamador pode passar aqui
    tensores 4D que este otimizador achata e desachata internamente."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                         ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=1 - momentum)

                orig_shape = buf.shape
                if buf.ndim == 4:
                    # kernel de conv: (C_out, C_in, k_h, k_w) -> (C_out, C_in*k_h*k_w)
                    mat = buf.reshape(orig_shape[0], -1)
                elif buf.ndim == 2:
                    mat = buf
                else:
                    # vieses / vetores 1-D não deveriam estar aqui, mas por
                    # segurança caímos para um passo de SGD com momento puro
                    p.add_(buf, alpha=-lr)
                    continue

                ortho = newton_schulz_ortho(mat, steps=ns_steps)
                update = ortho.reshape(orig_shape)
                p.add_(update, alpha=-lr)

        return loss
