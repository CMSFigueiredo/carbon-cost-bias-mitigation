import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_LABEL = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B-Instruct-4bit": "Qwen2.5-7B-4bit",
    "Qwen/Qwen2.5-7B-Instruct-8bit": "Qwen2.5-7B-8bit",
    "openai/gpt-3.5-turbo": "GPT-3.5-turbo",
    "openai/gpt-5-mini": "GPT-5-mini",
}
TECH_SHORT = {
    "baseline": "baseline", "zeroshot_instruction": "zero-shot",
    "reprompting": "reprompting", "self_debias_explanation": "explanation",
}
MEASURED = {"Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct-4bit", "Qwen/Qwen2.5-7B-Instruct-8bit"}

def bias_mag(amb, dis):
    return (abs(amb) + abs(dis)) / 2

# Overrides p/ pontos onde out/results.csv ficou temporariamente
# dessincronizado da Tabela 5 do artigo (reexecuções pontuais de
# baseline/zero-shot/explicação do 8-bit sobrescreveram os runs originais
# usados na tabela -- ver notas de rodapé da Tabela 5 no artigo). Estes
# valores são os mesmos reportados na Tabela 5 (gco2_mean em gCO2); a
# figura deve ficar consistente com o texto, não com o estado transitório
# do CSV. Remover a entrada aqui quando os dados subjacentes forem
# re-sincronizados com a tabela.
TABLE5_OVERRIDES = {
    ("Qwen/Qwen2.5-7B-Instruct-8bit", "baseline"): {"gco2_mean": 2.028, "bias_mag": 0.0278},
    ("Qwen/Qwen2.5-7B-Instruct-8bit", "zeroshot_instruction"): {"gco2_mean": 2.066, "bias_mag": 0.0211},
    ("Qwen/Qwen2.5-7B-Instruct-8bit", "self_debias_explanation"): {"gco2_mean": 36.00, "bias_mag": 0.0138},
}

ap = argparse.ArgumentParser()
ap.add_argument("--local-results", type=Path, default=Path("../results/local/results.csv"))
ap.add_argument("--api-results", type=Path, default=Path("../results/api/results_api.csv"))
ap.add_argument("--out", type=Path, default=Path("../results/pareto_todos_modelos.png"))
args = ap.parse_args()

r_local = pd.read_csv(args.local_results)
r_api = pd.read_csv(args.api_results)
df = pd.concat([r_local, r_api], ignore_index=True)

agg = df.groupby(["model", "technique"]).agg(
    gco2_mean=("gco2", "mean"),
    bias_ambig=("bias_ambig", "mean"),
    bias_disambig=("bias_disambig", "mean"),
).reset_index()
agg["bias_mag"] = agg.apply(lambda r: bias_mag(r["bias_ambig"], r["bias_disambig"]), axis=1)

for (model, tech), vals in TABLE5_OVERRIDES.items():
    mask = (agg["model"] == model) & (agg["technique"] == tech)
    if mask.any():
        for col, v in vals.items():
            agg.loc[mask, col] = v
    else:
        row = {"model": model, "technique": tech, "bias_ambig": float("nan"), "bias_disambig": float("nan")}
        row.update(vals)
        agg = pd.concat([agg, pd.DataFrame([row])], ignore_index=True)

fig, ax = plt.subplots(figsize=(8.5, 6))
x_all = agg["gco2_mean"].to_numpy()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
model_order = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct-4bit",
               "Qwen/Qwen2.5-7B-Instruct-8bit", "openai/gpt-3.5-turbo", "openai/gpt-5-mini"]

for i, model in enumerate(model_order):
    g = agg[agg["model"] == model]
    measured = model in MEASURED
    marker = "o" if measured else "^"
    facecolor = colors[i] if measured else "none"
    ax.scatter(g["gco2_mean"], g["bias_mag"], label=MODEL_LABEL[model], zorder=3, s=45,
               marker=marker, facecolors=facecolor, edgecolors=colors[i], linewidths=1.5)
    for _, row in g.iterrows():
        ax.annotate(TECH_SHORT.get(row["technique"], row["technique"]),
                    (row["gco2_mean"], row["bias_mag"]),
                    fontsize=7, xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center")

ax.set_xlim(-2, 60)
ax.set_xlabel("Emissions per full pass (gCO2, linear scale) — lower is better")
ax.set_ylabel("bias_mag = (|s_AMB| + |s_DIS|) / 2 — lower is better")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"[info] figura salva em {args.out}")
