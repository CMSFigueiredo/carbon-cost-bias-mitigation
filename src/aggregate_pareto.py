"""
Agregação + fronteira de Pareto (viés x carbono) a partir do results.csv.

Produz:
  - summary.csv : média/desvio por (modelo, técnica) de acurácia, energia e
    viés, mais o "custo de equidade" = ΔgCO2 / Δ|viés| relativo ao baseline.
  - pareto.png  : |viés| (menor = melhor) x gCO2 (menor = melhor), por modelo.

Uso:
    python aggregate_pareto.py --in results.csv --summary summary.csv --plot pareto.png

Notas:
  - Δ é sempre relativo ao baseline do MESMO modelo (a comparação justa).
  - viés é combinado em magnitude |s_amb| e |s_dis|; separe se preferir.
  - custo de equidade só é definido quando a técnica REDUZ viés (Δ|viés| > 0);
    caso contrário fica NaN (a técnica gastou energia sem melhorar equidade —
    um achado por si só).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def bias_magnitude(bias_ambig, bias_disambig) -> float:
    vals = [abs(v) for v in (bias_ambig, bias_disambig) if pd.notna(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "acc_ambig", "acc_disambig", "kwh", "gco2",
               "bias_ambig", "bias_disambig", "unparseable"]
    agg = df.groupby(["model", "technique"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["bias_mag"] = agg.apply(
        lambda r: bias_magnitude(r["bias_ambig_mean"], r["bias_disambig_mean"]), axis=1)
    return agg


def equity_cost(agg: pd.DataFrame) -> pd.DataFrame:
    """Custo de equidade relativo ao baseline, por modelo."""
    out = []
    for _, g in agg.groupby("model"):
        base = g[g["technique"] == "baseline"]
        if base.empty:
            print(f"[aviso] sem baseline para {g['model'].iloc[0]}; pulando")
            continue
        b_bias = base["bias_mag"].iloc[0]
        b_gco2 = base["gco2_mean"].iloc[0]
        for _, r in g.iterrows():
            d_bias = b_bias - r["bias_mag"]        # >0 = reduziu viés
            d_gco2 = r["gco2_mean"] - b_gco2       # custo extra de carbono
            cost = d_gco2 / d_bias if d_bias > 1e-9 else float("nan")
            row = r.to_dict()
            row.update(delta_bias=d_bias, delta_gco2=d_gco2,
                       custo_equidade_gco2_por_ponto=cost)
            out.append(row)
    return pd.DataFrame(out)


def plot_pareto(summary: pd.DataFrame, out_plot: Path):
    if not _HAS_MPL:
        print("[aviso] matplotlib ausente; pulei a figura")
        return
    import itertools
    from matplotlib.ticker import FuncFormatter, LogLocator

    # eixo x em log: gco2_mean cobre quase 2 ordens de grandeza (baseline
    # barato x self-debias_explanation caro) -- em escala linear os pontos
    # baratos colapsam num aglomerado ilegível perto da origem.
    fig, ax = plt.subplots(figsize=(8.5, 6))
    x_all = summary["gco2_mean"].to_numpy()
    x_mid = (x_all.min() * x_all.max()) ** 0.5  # centro geométrico (escala log)
    # ciclo de deslocamentos verticais só p/ espalhar rótulos próximos no
    # mesmo lado; a direção horizontal é decidida pela posição do ponto
    # (direita do centro -> rótulo para a esquerda, e vice-versa) para
    # nunca invadir a legenda, ancorada fora da área do gráfico.
    y_cycle = itertools.cycle([8, -12, 16, -16])
    tech_short = {
        "baseline": "baseline", "zeroshot_instruction": "zero-shot",
        "reprompting": "reprompting", "self_debias_explanation": "explicação",
    }
    for model, g in summary.groupby("model"):
        ax.scatter(g["gco2_mean"], g["bias_mag"], label=model, zorder=3, s=35)
        for _, r in g.iterrows():
            dy = next(y_cycle)
            if r["gco2_mean"] >= x_mid:
                dx, ha = -6, "right"
            else:
                dx, ha = 6, "left"
            ax.annotate(tech_short.get(r["technique"], r["technique"]),
                        (r["gco2_mean"], r["bias_mag"]),
                        fontsize=7, xytext=(dx, dy), textcoords="offset points",
                        ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(x_all.min() / 2, x_all.max() * 2.2)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(0.1, 0.2, 0.5, 1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Emissões por passada (gCO2, escala log)  —  menor é melhor")
    ax.set_ylabel("|viés|  —  0 = sem viés")
    ax.set_title("Fronteira viés x carbono por técnica")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"[info] figura salva em {out_plot}")


def main(in_csv: Path, out_summary: Path, out_plot: Path):
    df = pd.read_csv(in_csv)
    agg = aggregate(df)
    summary = equity_cost(agg)
    summary.to_csv(out_summary, index=False)

    cols = ["model", "technique", "accuracy_mean", "gco2_mean", "bias_mag",
            "delta_bias", "custo_equidade_gco2_por_ponto"]
    print("\n=== Resumo: acurácia, carbono, viés e custo de equidade ===")
    print(summary[cols].to_string(index=False))
    print(f"\n[info] resumo salvo em {out_summary}")

    plot_pareto(summary, out_plot)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", type=Path, default=Path("results.csv"))
    ap.add_argument("--summary", type=Path, default=Path("summary.csv"))
    ap.add_argument("--plot", type=Path, default=Path("pareto.png"))
    a = ap.parse_args()
    main(a.in_csv, a.summary, a.plot)
