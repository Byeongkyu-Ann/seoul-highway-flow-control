# NoJam — Macroscopic Traffic Flow Control for Seoul's Urban Highways

**English** | [한국어](README.ko.md)

An engineering-design-contest project ("설계아이디어 부문") that predicts traffic breakdown on
Seoul's eight urban highways before it happens and reroutes vehicles network-wide to prevent it,
instead of reacting after congestion has already formed.

## Overview

Existing navigation apps (T-map, Kakao Navi) and public ITS detect congestion after it has already
happened and route individual drivers around it. When many drivers are sent to the same detour, that
detour becomes the next bottleneck — the congestion is relocated, not resolved.

This project proposes **macroscopic flow control**: forecast the near-future state of every highway
link, convert that forecast into a time-varying travel cost, and spread vehicles across several routes
before a jam forms.

```
historical traffic + weather + accident-rate + holiday data
            |
            v
  LSTM + Transformer ensemble  -> VOL / SPD / OCC forecast, 3 steps (15 min) ahead
            |
            v
  BPR cost function  -> per-edge travel cost from the forecast
            |
            v
  Yen's K-shortest paths + softmax multi-vehicle assignment + congestion feedback
            |
            v
  Dijkstra, re-run every 5 minutes  -> routes shown in a PyQt5 GUI
```

The test network is a 9-node, 20-edge subgraph of the Gangbuk-side ring of highways (Gangbyeonbuk-ro,
Naebu Expressway, Dongbu and Bukbu Expressways), chosen for its loop/grid topology, which gives Yen's
algorithm physically distinct alternative routes to choose from.

## Repository structure

```
notebooks/
  01_data_preprocessing.ipynb   raw VDS logs -> one model-ready feature table
  02_ensemble_comparison.ipynb  4 ways to combine the LSTM and Transformer forecasts
  03_routing_and_gui.ipynb      live weather, BPR cost, Dijkstra rerouting, PyQt5 app
```

Notebooks `02` and `03` load checkpoints produced by LSTM and Transformer training notebooks that were
not included in what was handed over for this writeup — see **What is missing** below.

## Data

The dataset is 5-minute traffic-sensor readings for 160 links of the Seoul urban highway network,
collected over 2025. 160 per-link CSV files were merged into **16,491,792 records across 25 variables**,
covering 16 routes (`ROUTE_ID`) and 160 links (`LINK_ID`).

| Field | Meaning | Preprocessing |
|---|---|---|
| `ROUTE_ID` (16) | route identifier | label encoding -> 8-dim embedding |
| `LINK_ID` (160) | link identifier | label encoding -> 50-dim embedding |
| `VOL` | traffic volume / 5 min | log1p + standardize. Target and autoregressive input |
| `SPD` | speed (km/h) | standardize. Target and autoregressive input |
| `OCC` | occupancy (%) | log1p + standardize. Target and autoregressive input |
| `day_sin/cos`, `hour_sin/cos` | time of year / day of day | sin-cos encoding, used raw |
| `Past SPD` | previous link's speed | standardize |
| `acc_weekday/month/hour` | accident rate by weekday, month, hour | min-max scaled |
| rainfall (mm), snowfall (cm) | weather | log1p + min-max |
| visibility (10 m) | weather | log(1+x) |
| `is_holiday` | public holiday | 0/1 |
| weekday (`wd_0`..`wd_6`) | day of week | one-hot (7-dim) |

Each of the 160 links is sequenced independently with a sliding window of **`SEQ_LEN=12`** (1 hour of
history) predicting **`PRED_STEPS=3`** (15 minutes ahead), giving **16,489,552** samples. The split is
time-ordered — 70% train / 15% validation / 15% test — to prevent future data from leaking into training.

## Forecasting models

Two base learners, trained independently, feed a stacking ensemble.

| | LSTM | Transformer |
|---|---|---|
| Role | short-term continuity, gated memory | long-range / periodic pattern via self-attention |
| Parameters | 5,460,553 | 292,553 |
| Architecture (after Bayesian optimization) | 1 layer, hidden 512 | 2 encoder layers, `d_model=256`, 4 heads, pre-LN |
| Loss | Peak-Weighted MSE: `L = mean[(1 + α·|y_true|)·(y_pred − y_true)²]`, α=0.5 | same |
| Optimizer / schedule | AdamW, linear warmup + cosine annealing, gradient clipping | same |
| HP search | Optuna TPE, 30 trials, best val loss 0.0490 | Optuna TPE + Hyperband, 30 trials, best val loss 0.0389 |

**Test set results (standalone models):**

| Target | LSTM RMSE | LSTM R² | Transformer RMSE | Transformer R² |
|---|---|---|---|---|
| VOL (volume) | 25.07 | 0.9364 | 23.37 | 0.9447 |
| SPD (speed, km/h) | 5.99 | 0.8775 | 5.53 | 0.8939 |
| OCC (occupancy) | 3.64 | 0.8779 | 3.25 | 0.9002 |
| **mean** | – | 0.8973 | – | 0.9129 |

Four stacking strategies were compared on a held-out set of predictions: a Bayesian-optimized linear
weight, XGBoost, Random Forest, and Ridge regression, each taking the 18-dim concatenation of the two
base learners' outputs as input. **No single method won on every target**, so the final system uses a
per-target ensemble:

| Target | Best method | R² | vs. best standalone model |
|---|---|---|---|
| VOL | BO-weighted average (Transformer weight ≈ 0.86–0.91) | 0.9453 | +0.07 pp |
| SPD | Ridge stacking | 0.8972 | +0.33 pp |
| OCC | Ridge stacking | 0.9082 | +0.80 pp |

Ridge outperforming the nonlinear stackers (XGBoost, Random Forest) on SPD/OCC was explained by the
report as: the two base learners' predictions are already close to a linear relationship with the
target, so a nonlinear second stage mostly overfits.

## Routing layer

- **Edge cost**: the BPR function `t = t₀ · [1 + α·(V/C)^β]`, α=0.15, β=4.0. V/C is taken as the larger
  of `OCC/100` and `VOL/LINK_CAPACITY`, a conservative choice that uses whichever signal shows more
  congestion.
- **Multi-vehicle assignment**: Yen's K-shortest paths (K=3) finds route candidates; a softmax over
  their costs (temperature T=2.5) spreads a batch of vehicles across them instead of sending all of
  them down the single shortest path; assigned vehicles feed back into the edge's VOL, which raises its
  cost for the next batch.
- **Re-routing**: Dijkstra re-runs on the updated graph every 5 minutes.
- **Live inputs**: current rainfall, snowfall and visibility are scraped from the Korea Meteorological
  Administration site (`weather.go.kr`) at request time, with the previous value kept if a scrape fails.
- **Interface**: a PyQt5 desktop app takes an origin and destination and displays the recommended route.

## Results

- Standalone and ensemble forecast accuracy: see the tables above (R² 0.90–0.95 across VOL/SPD/OCC on
  the test set).
- System-level routing result reported in the proposal abstract: an **8.5% reduction in total network
  cost** compared to the baseline of sending every vehicle down the same shortest path, attributed to
  the softmax-based multi-vehicle spreading that avoids turning the alternative route into a new
  bottleneck.
- The report presents this on the 9-node Gangbuk-side test network. It was not run on the full 8-highway
  Seoul network.

## What is missing from this repository

The handoff did not include the notebooks that train the standalone LSTM and Transformer forecasters
(`02_ensemble_comparison.ipynb` and `03_routing_and_gui.ipynb` both load `best_lstm_model.pt`,
`best_transformer_model.pt`, `scalers.pkl` and related files, which are produced by notebooks not
present here). Without them, the pipeline cannot be re-run end to end from this repository alone.

Known rough edges, left as-is rather than silently fixed:
- The notebooks contain hardcoded Windows paths (`D:\...`) and depend on a CUDA GPU (`device='cuda'`,
  `torch.cuda.amp.autocast`). They will need path and device edits to run elsewhere.
- `02_ensemble_comparison.ipynb` has one duplicated cell near the end (saving the BO weights and the
  Ridge model is done twice, in cells under "11. 최종 결과 저장" and again under "시간 아끼려고 임시로
  코드 작성").
- `03_routing_and_gui.ipynb`'s holiday table is hardcoded for 2026 and will need updating for other years.
- None of the three notebooks were re-executed for this writeup; they require the private dataset,
  trained checkpoints and a GPU that were not part of the handoff.

## Project information

- Team: NoJam
- Category: 설계아이디어 (Design Idea) track, selected topic (A) 인구구조 변화 대응 및 스마트 도시·산업 전환
- Stack: Python, PyTorch, Optuna, XGBoost, scikit-learn, PyQt5, Selenium/BeautifulSoup (weather scraping)
