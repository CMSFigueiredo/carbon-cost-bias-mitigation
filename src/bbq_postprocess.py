"""
Pós-processamento de predições já coletadas -- sem GPU, sem recarregar
modelo, sem inferência.

Por que isso existe separado do harness: bbq_energy_harness.py já calcula
`preds` completos em toda execução (inclusive a primeira, run=0) de cada
(modelo, técnica), só para medir energia. Como a decodificação é gulosa e
determinística, esse `preds` é idêntico em todas as N execuções (confirmado
empiricamente: desvio padrão zero em acurácia e viés). Antes, um segundo
script (bbq_category_breakdown.py) recarregava o modelo e repetia a
inferência do zero só para ter esse mesmo `preds` -- redundância pura, e
fonte de um bug real (resultado obsoleto por divergência entre as duas
passadas). Aqui, o harness passa o `preds` de run=0 direto pra estas
funções, sem custo de GPU extra.

Não importa nada de bbq_energy_harness.py (evita import circular, já que é
o harness quem importa este módulo).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bbq_bias import LETTERS, accuracy, biased_index, compute_bias_scores


def _letter(idx):
    return LETTERS[idx] if idx is not None else None


def dump_predictions(model_id, items, tech_name, preds) -> list[dict]:
    """Formato longo: 1 linha por (item, técnica) -- auditoria/reprodutibilidade."""
    rows = []
    for idx, (it, p) in enumerate(zip(items, preds)):
        b = biased_index(it)
        rows.append({
            "model": model_id,
            "technique": tech_name,
            "item_idx": idx,
            "category": it.category,
            "context_condition": it.context_condition,
            "question_polarity": it.question_polarity,
            "correct_idx": it.correct_idx,
            "pred": p,
            "biased_idx": b,
            "is_biased_answer": (p is not None and b is not None and p == b),
        })
    return rows


def category_summary(model_id, tech_name, items, preds) -> list[dict]:
    """bias_ambig, bias_disambig, acurácias e n_bias_resolved por categoria."""
    rows = []
    categories = sorted({it.category for it in items})
    for cat in categories:
        cat_items, cat_preds = [], []
        for it, p in zip(items, preds):
            if it.category == cat:
                cat_items.append(it)
                cat_preds.append(p)
        bias = compute_bias_scores(cat_preds, cat_items)
        rows.append({
            "model": model_id,
            "technique": tech_name,
            "category": cat,
            "n_items": len(cat_items),
            "accuracy": accuracy(cat_preds, cat_items),
            "bias_ambig": bias["bias_ambig"],
            "bias_disambig": bias["bias_disambig"],
            "acc_ambig": bias["acc_ambig"],
            "acc_disambig": bias["acc_disambig"],
            "n_bias_resolved": bias["n_bias_resolved"],
        })
    return rows


def dump_raw_responses(model_id, tech_name, texts, run: int = 0) -> list[dict]:
    """Formato longo: 1 linha por item -- texto bruto gerado pelo modelo,
    antes de qualquer parsing. Existe pra permitir auditoria futura (ex.:
    validar o parser, investigar formatação) sem precisar re-rodar
    inferência/chamadas de API."""
    return [{
        "model": model_id,
        "technique": tech_name,
        "run": run,
        "item_idx": idx,
        "raw_text": t,
    } for idx, t in enumerate(texts)]


def save_item_results_wide(model_id: str, items, tech_preds: dict[str, list], path: Path) -> None:
    """1 linha por pergunta, 1 coluna por técnica -- rastreabilidade
    individual: dá pra abrir e conferir exatamente o que o modelo respondeu
    em cada prompt, item a item.

    Se `path` já existir e `tech_preds` cobrir só um subconjunto das
    técnicas (rodada incremental por técnica, ex.: só reprompting), as
    colunas das técnicas ausentes em `tech_preds` são preservadas do
    arquivo antigo em vez de descartadas -- assume mesma amostra/ordem de
    itens (mesmo `sampled_dataset.csv`), verificado pelo tamanho."""
    rows = []
    for idx, it in enumerate(items):
        row = {
            "item_id": idx,
            "context": it.context,
            "question": it.question,
            "context_condition": it.context_condition,
            "category": it.category,
            "correct_answer": _letter(it.correct_idx),
        }
        for tech_name, preds in tech_preds.items():
            row[tech_name] = _letter(preds[idx])
        rows.append(row)
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        if len(old_df) == len(new_df):
            for col in old_df.columns:
                if col not in new_df.columns:
                    new_df[col] = old_df[col]
        else:
            print(f"[aviso] {path} tinha {len(old_df)} linhas, amostra atual "
                  f"tem {len(new_df)} -- não deu pra preservar colunas antigas "
                  "(amostra parece ter mudado); arquivo reescrito só com as "
                  "técnicas desta rodada.")
    new_df.to_csv(path, index=False)
    print(f"[info] respostas por item (formato largo) salvas em {path}")


def _load_prior(path: Path, models: list[str], techniques: list[str] | None = None) -> list[dict]:
    """Preserva linhas de OUTROS modelos já salvos (rodadas incrementais);
    descarta linhas antigas do(s) modelo(s) que vão ser escritos agora. Se
    `techniques` for informado, a descarta é mais granular: só remove
    linhas do modelo QUE TAMBÉM sejam de uma dessas técnicas -- permite
    rodar de novo só uma técnica sem apagar as outras já salvas do mesmo
    modelo."""
    if not path.exists():
        return []
    df = pd.read_csv(path)
    mask = df["model"].isin(models)
    if techniques is not None:
        mask &= df["technique"].isin(techniques)
    return df[~mask].to_dict("records")


def write_postprocess_outputs(model_id: str, items, tech_preds: dict[str, list],
                               preds_out: Path, category_out: Path,
                               item_results_dir: Path,
                               tech_texts: dict[str, list] | None = None,
                               raw_out: Path | None = None) -> None:
    """Escreve predictions.csv, category_breakdown.csv e
    item_results_<modelo>.csv para UM modelo, a partir de preds já
    calculados (sem GPU) -- preserva o que já estava salvo de outros
    modelos, e de outras técnicas do MESMO modelo não recalculadas nesta
    chamada (`tech_preds` cobrindo só um subconjunto = rodada incremental
    por técnica). Se `tech_texts`/`raw_out` forem passados, escreve também
    o texto bruto (run=0 -- decodificação local é determinística, run=0
    representa qualquer execução)."""
    techs = list(tech_preds.keys())
    preds_rows = _load_prior(preds_out, [model_id], techs)
    summary_rows = _load_prior(category_out, [model_id], techs)
    for tech_name, preds in tech_preds.items():
        preds_rows.extend(dump_predictions(model_id, items, tech_name, preds))
        summary_rows.extend(category_summary(model_id, tech_name, items, preds))

    pd.DataFrame(preds_rows).to_csv(preds_out, index=False)
    print(f"[info] predições por item salvas em {preds_out}")
    pd.DataFrame(summary_rows).to_csv(category_out, index=False)
    print(f"[info] resumo por categoria salvo em {category_out}")

    slug = model_id.replace("/", "_")
    save_item_results_wide(model_id, items, tech_preds,
                            item_results_dir / f"item_results_{slug}.csv")

    if tech_texts is not None and raw_out is not None:
        raw_rows = _load_prior(raw_out, [model_id], list(tech_texts.keys()))
        for tech_name, texts in tech_texts.items():
            raw_rows.extend(dump_raw_responses(model_id, tech_name, texts, run=0))
        pd.DataFrame(raw_rows).to_csv(raw_out, index=False)
        print(f"[info] texto bruto (run=0) salvo em {raw_out}")
