# The Carbon Cost of Bias Mitigation

Reproducibility package for the paper *"The Carbon Cost of Bias Mitigation: Fairness–Sustainability Trade-offs in LLM Prompting"*.

It contains the evaluation harnesses used to measure accuracy, bias, energy consumption (local models) and estimated energy/cost (closed models via API) on the [BBQ benchmark](https://github.com/nyu-mll/BBQ), plus the aggregated results reported in the paper.

## What is (and isn't) in this repository

- `src/` — the full pipeline source code.
- `results/` — the **aggregated, numeric** results produced by the pipeline (accuracy, bias scores, energy/emissions, token counts). These contain no text from the BBQ dataset.
- The BBQ dataset itself is **not** redistributed here. It is third-party material (CC BY 4.0, [nyu-mll/BBQ](https://github.com/nyu-mll/BBQ)); see [Getting the dataset](#getting-the-dataset) below for how to obtain it. Per-item files that embed the original question/context text (e.g. `item_results_<model>.csv`, `sampled_dataset.csv`) are likewise not included — they are regenerated deterministically by running the pipeline (see [Reproducing the exact sample](#reproducing-the-exact-sample)).

## Repository structure

```
.
├── requirements.txt
├── .env.example
├── src/
│   ├── bbq_energy_harness.py    # main entry point: local (open) models, measured energy via CodeCarbon
│   ├── bbq_api_harness.py       # closed models via OpenRouter API, estimated energy via EcoLogits
│   ├── bbq_bias.py              # bias-score computation shared by both harnesses
│   ├── bbq_postprocess.py       # per-item / per-category output generation
│   ├── aggregate_pareto.py      # aggregation + Pareto-frontier plot (called by bbq_energy_harness.py)
│   ├── bbq_token_audit.py       # token-usage audit (local + API) used for the token-count table
│   └── plot_pareto_5models.py   # combined 5-model (local + API) Pareto figure used in the paper
└── results/
    ├── local/    # open models (Qwen2.5-1.5B, Qwen2.5-7B 4-bit, Qwen2.5-7B 8-bit)
    └── api/      # closed models (GPT-3.5-turbo, GPT-5-mini)
```

## Environment setup

Python 3.13. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- Running `bbq_energy_harness.py` (open models) requires a CUDA GPU for realistic energy measurements via [CodeCarbon](https://github.com/mlco2/codecarbon); `bitsandbytes` is only needed for the 4-bit/8-bit quantized conditions.
- Running `bbq_api_harness.py` (closed models) requires an [OpenRouter](https://openrouter.ai/) API key. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`.

## Getting the dataset

Clone the official BBQ repository anywhere on disk:

```bash
git clone https://github.com/nyu-mll/BBQ.git
```

The harnesses expect `--bbq-dir` to point at `BBQ/data` (the `.jsonl` files), and read `additional_metadata.csv` from the sibling `BBQ/analysis_scripts/` directory automatically.

## Reproducing the exact sample

The harness draws a stratified sample (500 items per category × 9 categories = 4,500 items) with a fixed seed, so re-running it against the official BBQ data deterministically reproduces the exact same sample used in the paper:

- Categories: `Age, Gender_identity, Race_ethnicity, Religion, Nationality, Disability_status, Physical_appearance, SES, Sexual_orientation`
- Items per category: `500`
- Seed: `42`

The sample is cached to `--sample-path` on first run and reused on subsequent runs (including by `bbq_api_harness.py`, so local and API models are evaluated on the identical item set).

## Running the pipeline

All commands below assume `src/` as the working directory and an `BBQ_DIR` variable pointing at your BBQ clone's `data/` folder.

### 1. Open models (measured energy)

```bash
cd src
python bbq_energy_harness.py \
  --bbq-dir "$BBQ_DIR" \
  --out ../results/local/results.csv \
  --sample-path ../results/local/sampled_dataset.csv \
  --preds-out ../results/local/predictions.csv \
  --category-out ../results/local/category_breakdown.csv \
  --summary-out ../results/local/summary.csv \
  --pareto-out ../results/local/pareto.png \
  --raw-out ../results/local/raw_responses.csv
```

Optional flags: `--models` (comma-separated subset), `--techniques` (comma-separated subset of `baseline,zeroshot_instruction,self_debias_explanation,reprompting`), `--n-runs` (repetitions per condition; default 3, for energy-noise averaging).

### 2. Closed models via API

```bash
cd src
python bbq_api_harness.py --dry-run          # estimate cost only, no calls made
python bbq_api_harness.py --budget-usd 20    # run for real, stop before exceeding the budget
```

Reads the same cached sample from `../results/local/sampled_dataset.csv`; writes to `../results/api/`.

### 3. Token-usage audit (optional)

```bash
cd src
python bbq_token_audit.py
```

### 4. Combined 5-model Pareto figure

```bash
cd src
python plot_pareto_5models.py
```

Reads `results/local/results.csv` and `results/api/results_api.csv`, writes `results/pareto_todos_modelos.png`.

## Output columns

- `results.csv` / `results_api.csv`: one row per (model, technique, run) — accuracy, bias scores (ambiguous/disambiguated), unparseable rate, energy (kWh/gCO2 — measured for local, EcoLogits-estimated for API), and (API only) token counts and real USD cost.
- `category_breakdown*.csv`: the same metrics broken down by BBQ category.
- `predictions*.csv`: per-item predicted answer index, correct/biased answer index — no question/context text.
- `raw_responses*.csv`: raw model output per item/run (the model's answer letter or text) — no BBQ question/context text.
- `summary*.csv`: per (model, technique) means/standard deviations across runs, plus the derived *equity cost* metric (gCO2 per point of bias reduced) used in the paper's Pareto analysis.
