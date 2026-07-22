# Stock Investment Screening Research

Personal research notebook that screens public companies using **fundamentals** plus a qualitative business-outlook signal, then applies explicit investment rules.

## Project overview

Exploratory investment analysis combining tabular fundamentals with scraped qualitative indicators and simple statistical checks (including linear regression sketches).

## Business problem

Investors face too many tickers. A transparent, rule-based screen—documented and repeatable—helps focus diligence on a smaller set of candidates.

## Solution

1. Maintain ticker universes / intermediate tables under `data/`.
2. Enrich with publicly available fundamentals and outlook-style signals.
3. Apply documented inclusion/exclusion rules in the notebook.
4. Review distributions and simple model fits as research aids—not production trading signals.

## Architecture

```mermaid
flowchart LR
  Universe[Ticker_Universe_CSV] --> Enrich[Public_Data_Enrichment]
  Enrich --> Rules[Rule_Based_Screen]
  Rules --> Shortlist[Candidate_List]
  Shortlist --> Review[Manual_Diligence]
```

## ML / analytics methodology

- Descriptive statistics and filters on fundamentals
- Optional `LinearRegression` sketches for research relationships
- **Not** a backtested trading strategy; no performance claims

## Repository layout

```
investment_strategy_Python/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                 # Monthly / intermediate CSVs
├── notebooks/
│   └── Stock_Investment_Analysis.ipynb
├── docs/archive/         # Merged artifacts from former Stock_Python
│   ├── stock_code_example.html
│   └── Stock_Python_README.md
└── tests/
```

Browser driver binaries are **not** stored in this repo (see `.gitignore`). Install drivers locally if you re-run scrape cells.

## Merge history

Former `Stock_Python` (redacted scraper HTML export + notes) was merged into `docs/archive/`. Prefer the research notebook under `notebooks/` as the active work product.


## Technologies

Python 3 · pandas · NumPy · matplotlib · scikit-learn · Selenium / BeautifulSoup (optional scrape cells) · Jupyter

## Installation

```bash
cd investment_strategy_Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook notebooks/Stock_Investment_Analysis.ipynb
```

Ensure paths resolve to `../data/...` (already updated for the `notebooks/` layout).

```bash
pytest -q
```

## Example outputs

- Filtered ticker tables for a given month
- Scatter/regression diagnostic plots for research hypotheses

## Product decisions and tradeoffs

- Rules are explicit and auditable; they are not optimized for historical Sharpe.
- Scraping cells may break when third-party sites change HTML—treat as optional.
- Respect site terms of service; prefer official APIs for any production system.

## Future improvements

- Replace brittle HTML scraping with licensed fundamentals APIs
- Add walk-forward evaluation if promoting rules beyond research
- Containerize a headless refresh job with clear rate limits

## Disclaimer

Educational / personal research only. Not investment advice.

## License

All Rights Reserved. See [LICENSE](LICENSE).
