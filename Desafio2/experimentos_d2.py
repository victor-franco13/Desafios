"""
experimentos_d2.py — compara configurações de desafio2_CaioCouto_VictorFranco.py usando o próprio harness.

Uso (rodar de dentro da pasta Desafio2):
    python experimentos_d2.py                                   # triagem: 25% dos dados, 1 época, 1 semente
    python experimentos_d2.py --fracao 1 --epocas 3 --sementes 3 --candidatos relu_ort_s01 gelu_grad_ort
    python experimentos_d2.py --listar

Baseline e referência de cada modo (fracao/epocas) ficam no cache calibracao_d2.json do harness.
Resultados vão sendo gravados em resultados_d2.json.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import harness_desafio2 as H

AQUI = os.path.dirname(os.path.abspath(__file__))
sub = H.carregar_submissao(os.path.join(AQUI, "desafio2_CaioCouto_VictorFranco.py"))

B = dict(alfa=0.2, ganho=1.0, criterio="variancia", pesos="ortogonal", escala_saida=0.1)
CANDIDATOS = {
    # sanidade: deve reproduzir a referência (s_t ≈ 1)
    "he_ref":          {**B, "ativ": "relu", "pesos": "normal", "escala_saida": 1.0},
    # o que muda em cima de ReLU + He
    "relu_s01":        {**B, "ativ": "relu", "pesos": "normal"},
    "relu_ort":        {**B, "ativ": "relu", "escala_saida": 1.0},
    "relu_ort_s01":    {**B, "ativ": "relu"},
    "relu_ort_s0":     {**B, "ativ": "relu", "escala_saida": 0.0},
    # taxa de aprendizado efetiva das camadas ocultas (× ganho²)
    "relu_ort_g07":    {**B, "ativ": "relu", "ganho": 0.7},
    "relu_ort_g14":    {**B, "ativ": "relu", "ganho": 1.4},
    "relu_ort_g05":    {**B, "ativ": "relu", "ganho": 0.5},
    "relu_ort_g03":    {**B, "ativ": "relu", "ganho": 0.3},
    "lrelu02_ort_g05": {**B, "ativ": "lrelu", "alfa": 0.2, "ganho": 0.5},
    "gelu_grad_g05":   {**B, "ativ": "gelu", "criterio": "gradiente", "ganho": 0.5},
    "tanh_grad_g05":   {**B, "ativ": "tanh", "criterio": "gradiente", "ganho": 0.5},
    "tanh_grad_g07":   {**B, "ativ": "tanh", "criterio": "gradiente", "ganho": 0.7},
    # outras ativações, ganho calculado
    "lrelu02_ort":     {**B, "ativ": "lrelu", "alfa": 0.2},
    "lrelu04_ort":     {**B, "ativ": "lrelu", "alfa": 0.4},
    "gelu_var_ort":    {**B, "ativ": "gelu"},
    "gelu_grad_ort":   {**B, "ativ": "gelu", "criterio": "gradiente"},
    "silu_grad_ort":   {**B, "ativ": "silu", "criterio": "gradiente"},
    "elu_var_ort":     {**B, "ativ": "elu"},
    "selu_lecun":      {**B, "ativ": "selu", "pesos": "normal"},
    "tanh_grad_ort":   {**B, "ativ": "tanh", "criterio": "gradiente"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidatos", nargs="*", default=list(CANDIDATOS))
    ap.add_argument("--fracao", type=float, default=0.25)
    ap.add_argument("--epocas", type=int, default=1)
    ap.add_argument("--sementes", type=int, default=1)
    ap.add_argument("--dados", nargs="*", default=["mnist", "fashion", "cifar10"])
    ap.add_argument("--L", nargs="*", type=int, default=list(H.PROFUNDIDADES))
    ap.add_argument("--saida", default="resultados_d2.json")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar:
        for k, v in CANDIDATOS.items():
            print(f"{k:<16} {v}")
        return

    sementes = H.SEMENTES[:max(1, min(3, args.sementes))]
    tarefas = [H.Tarefa(f"{d}_L{L}", d, L) for d in args.dados for L in args.L]
    print(f"device={H.DEVICE}  fracao={args.fracao}  epocas={args.epocas}  sementes={len(sementes)}")
    calib = H.calibrar(tarefas, sementes, args.fracao, args.epocas)

    modo = f"f={args.fracao}|e={args.epocas}|s={len(sementes)}"
    todos = json.load(open(args.saida, encoding="utf-8")) if os.path.exists(args.saida) else {}
    placar = []
    for nome in args.candidatos:
        ativ, init = sub._fazer(CANDIDATOS[nome])
        t0 = time.perf_counter()
        S, linhas = H.avaliar(SimpleNamespace(ativacao=ativ, inicializar=init), tarefas, calib,
                              sementes, args.fracao, args.epocas, verboso=False)
        dt = time.perf_counter() - t0
        st = " ".join(f"{l['s_t']:4.2f}" for l in linhas)
        acc = " ".join(f"{l['acc']:.3f}" for l in linhas)
        print(f"{nome:<16} S={S:6.1f}  ({dt:5.0f}s)  s_t: {st}\n{'':16} acc: {acc}")
        placar.append((S, nome))
        todos.setdefault(modo, {})[nome] = dict(config=CANDIDATOS[nome], escore=S, tarefas=linhas)
        with open(args.saida, "w", encoding="utf-8") as fh:
            json.dump(todos, fh, indent=1, ensure_ascii=False)

    print("\ntarefas:", " ".join(t.nome for t in tarefas))
    print("ranking:")
    for S, nome in sorted(placar, reverse=True):
        print(f"  {S:6.1f}  {nome}")


if __name__ == "__main__":
    main()
