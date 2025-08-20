# StockPredictionPro
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advanced, production-style stock market intelligence system that combines regression and classification models to predict short-term price movements, generate trade signals, and visualize results through a polished, multipage Streamlit UI. Built for fast iteration by a 3-person team in 5–6 days with clean architecture, governance, and portfolio-ready presentation.

- Tech: Python, Streamlit, scikit-learn, Plotly, yfinance, FastAPI (optional service layer)

- Models: Linear, Multiple, Polynomial (with regularization), Logistic, SVM, Random Forest, optional Stacking

- Features: 20+ indicators, time-series cross-validation, walk-forward backtests, signals, exports, governance logs

- Packaging: Docker, optional executable (PyInstaller) that launches Streamlit directly

- Note: This app is for educational and demonstration purposes only. It is not financial advice.

------------

## Table of Contents

- Project Goals

- System Architecture

- Key Features

- Data Sources

- Models and Targets

- Evaluation and Backtesting

- Streamlit App Pages

- Governance and Observability

- Project Structure

- Quickstart (Local)

- Build and Run with Docker

- Build a Standalone Executable

- API Overview (Optional)

- Collaboration Workflow (3 Developers)

- Configuration

- Testing

- Roadmap (v2 Ideas)

- License

--------------

### Project Goals

- Build a credible market analytics app that compares classical regression approaches with indicators and classification models.

- Translate predictions into actionable signals with basic risk controls and backtesting.

- Demonstrate strong ML, systems thinking, and product craft through clean UX, governance, and docs.

- Ship a polished demo in 5–6 days, ready for portfolios and presentations.

------------

### System Architecture

#### High-level flow:

- Data Layer: Fetch OHLCV from free sources (yfinance primary; Alpha Vantage fallback via .env), validate, cache, and persist snapshots.

- Features Layer: Engineer indicators (trend, momentum, volatility, volume), create lag/interaction/polynomial features, guard leakage, and build supervised targets (regression and classification).

- Model Layer: Train regression (linear, multiple, polynomial+regularization) and classification (logistic, SVM, random forest; optional stacking), select via time-series CV, and persist artifacts with metadata.

- Evaluation Layer: Compute metrics (RMSE/MAE/MAPE/R²; accuracy/precision/recall/F1), generate signals, run walk-forward backtests with transaction costs, and produce reports.

- UI Layer: Multipage Streamlit app with interactive charts, parameter controls, model comparison, signals, backtests, exports, and governance.

- Optional API: Lightweight FastAPI to serve data/models/predictions for integration.

-------------

### Key Features

#### Data and Indicators

- Free, real-world data (Yahoo Finance) with caching and validation.

- 20+ technical indicators: SMA, EMA, MACD, RSI, Bollinger Bands, ATR, OBV, ROC, ADX, Stoch, MFI, Donchian, Keltner, etc.

- Lagged features and leakage guards to ensure correctness.

#### Modeling

- Regression: Linear baseline, Multiple (indicators), Polynomial with Ridge/ElasticNet (degree 2–3).

- Classification: Logistic, SVM, Random Forest; optional stacking meta-learner.

- Time-series CV (expanding windows), Optuna/Grid tuning, feature selection.

#### Signals and Trading

- Buy/Sell/Hold signals using model outputs, indicator confirmations, and confidence thresholds.

- Simple risk controls: ATR-based stops, take-profit, transaction costs in backtests.

#### Evaluation

- Metrics: RMSE/MAE/MAPE/R² (regression), Accuracy/Precision/Recall/F1/ROC-AUC (classification), Sharpe/Sortino/Max Drawdown/CAGR/Hit Rate (backtesting).

- Model comparison dashboards and exportable reports.

#### UX and Governance

- Multipage Streamlit with Plotly charts (candlestick, overlays, panels).

- Config-driven runs, run_id and audit trails, structured logs, reproducible outputs.

- Exports: predictions, signals, backtest results, and HTML/Markdown reports.

-----------

### Data Sources

- Primary: Yahoo Finance (yfinance) — free, no key.

- Fallback: Alpha Vantage (free API key via .env) for resiliency.

- Supported markets: US and NSE (e.g., AAPL, MSFT, INFY.NS, TCS.NS, RELIANCE.NS).

- Note: For unit tests, local CSV fixtures are used to avoid hitting APIs.

----------

### Models and Targets

- Targets

    - Regression: next-day close (t+1). Optional t+3 and t+5 horizons.

    - Classification: direction classes based on pct change thresholds:

        - 2-class: Up vs Down

        - 3-class: Up / Sideways / Down

        - Custom thresholds configurable in model_config.yaml.

    - Regression Models

        - Linear Regression (baseline).

        - Multiple Regression with indicators.

        - Polynomial Regression (degree 2–3) with Ridge/ElasticNet regularization.

        - Optional: SVR/RF as extra benchmarks if time permits.

    - Classification Models

        - Logistic Regression (interpretable baseline).

        - SVM (RBF) for non-linear decision boundaries.

        - Random Forest for robust non-parametric patterns and feature importance.

        - Optional: Stacking ensemble (meta-learner: Logistic).

    - Accuracy Strategy

        - Strong feature engineering: multi-timeframe indicators, lag features, regime tagging, feature scaling.

        - Regularization to reduce overfitting on polynomial features.

        - Optuna/Grid tuning on key hyperparameters (degree, alpha/C, class thresholds).

        - Time-series-aware validation and walk-forward testing.

--------------

### Evaluation and Backtesting

- Validation

    - TimeSeriesSplit (expanding windows), walk-forward evaluation for realistic performance.

    - Purged CV option (defer if time-constrained).

- Backtesting

    - Daily rebalancing, costs (bps), slippage parameter.

    - quity curve vs buy-and-hold, Sharpe/Sortino, Max Drawdown, Hit Rate, Trades log.

    - Position sizing: fixed or confidence-based; ATR stop-loss, take-profit.

---------------    

### Streamlit App Pages

- 01_Overview: Project summary, instructions, architecture, presets.

- 02_Data_&_Indicators: Select tickers, fetch/cache data, choose indicators, build features.

- 03_Chart_Analysis: Candlestick with overlays, RSI/MACD panels, time range controls.

- 04_Regression_Models: Train and compare linear/multiple/polynomial; metrics and plots.

- 05_Classification_Models: Train classifiers; confusion matrix, ROC, class metrics.

- 06_Model_Comparison: Side-by-side comparison across models and horizons.

- 07_Trading_Signals: Configure thresholds; view Buy/Sell/Hold markers and rules.

- 08_Portfolio_Management: Basic allocation/risk controls (optional v2).

- 09_Backtest_&_Performance: Equity curve, metrics, trades table, benchmark.

- 10_Exports_&_Reports: Download predictions/signals/backtests; auto-reports.

- 11_Admin_&_Logs: Config snapshot, logs tail, audit trail browser, run_id controls.

----------

### Governance and Observability

- Config-driven: All behavior controlled via YAML (app, model, indicators, trading).

- Audit trails: Each run emits metadata (config, params, tickers, time, git SHA).

- Logging: app.log, error.log, api.log, trading.log with structured entries.

- Reproducibility: Fixed seeds where applicable; saved artifacts with run_id.

---------------

### Project Structure

See full tree in the repository (StockPredictionPro/). Key directories:

- app/: Streamlit app, pages, components, styles, utilities.

- src/: core Python modules (data, features, models, evaluation, trading, api, utils).

- config/: YAML configurations.

- data/: raw/processed/models/predictions/backtests/exports.

- scripts/: setup, data, models, automation, deployment utilities.

- tests/: unit, integration, performance, fixtures.

- deployment/: docker, optional infra placeholders.

--------------

### Quickstart (Local)

#### Prerequisites

- Python 3.10+ recommended

- pip or conda

- Node not required

- Clone and install

    - git clone [(https://github.com/Faraazz05/StockPredictionPro.git)]

    - cd StockPredictionPro

    - python -m venv .venv && source .venv/bin/activate (Windows: .venv\Scripts\activate)

    - pip install -r requirements.txt

- Configure environment

    - cp .env.example .env

    - Optionally set ALPHA_VANTAGE_KEY in .env for fallback

    - Review config/app_config.yaml and model_config.yaml defaults

- Run Streamlit app

    - streamlit run app/streamlit_app.py

- First-run path in the UI:

    - Overview → Data & Indicators → Build features → Regression/Classification → Backtest → Exports.

-------------------    

### Build and Run with Docker

Build image

    - docker build -t stockpred-pro .

Run with docker-compose

    - docker-compose up --build

Access

    - Streamlit UI: http://localhost:8501

    - Optional API: http://localhost:8000/docs

-----------------    

### Build a Standalone Executable

#### For Windows/macOS/Linux, builds an executable that launches Streamlit directly.

Install build tools

    - pip install pyinstaller

Build

    - python build_exe.py

    - Output appears under dist/ (e.g., dist/StockPredPro.exe)

Note: On Windows, SmartScreen may warn on unsigned executables. Choose “Run anyway.”

--------------------------    

### API Overview (Optional)

#### If running the optional FastAPI service:

Key endpoints

    - GET /health/ping — service health check

    - GET /api/v1/data/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD

    - POST /api/v1/models/train — body includes model type, params, tickers, horizon

    - GET /api/v1/predictions/{run_id}/{symbol}

    - GET /api/v1/signals/{run_id}/{symbol}

    - POST /api/v1/backtests/run — backtest config in body

Security

    - Development uses open access. In production, enable auth/rate-limits and CORS policies.

-------------------    

### Collaboration Workflow (3 Developers)

Branching

    - main (release), develop (integration), feature/* for tasks, bugfix/* for fixes.

Process

    - Create issues with acceptance criteria; move via Project board (Backlog → In Progress → Review → Done).

Feature dev:

    - git checkout -b feature/short-name

Commit in small, logical increments with clear messages.

Open PR to develop; require review; squash merge.

Code quality

Use pre-commit (black, isort, flake8, YAML checks).

Unit tests run locally before PR; CI pipeline recommended.

Role split

Person A: Data & ML — data fetchers, indicators, regression/classification models, metrics, validation.

Person B: UI/UX & API — Streamlit pages/components, charts, optional API routes, exports.

Person C: Trading & Ops — signals, backtesting, Docker/compose, executable build, logging/governance.

---------------

### Configuration

Key files

    - config/app_config.yaml — UI defaults, paths, markets, cache settings.

    - config/model_config.yaml — model lists, CV folds, polynomial degree, regularization, thresholds.

    - config/indicators_config.yaml — indicator on/off and parameters.

    - config/trading_config.yaml — signal thresholds, transaction costs, stops/takes.

    - config/api_config.yaml — API toggles, rate limits (optional).

    - config/logging.yaml — logging handlers/formatters.

Environment (.env)

    - ALPHA_VANTAGE_KEY=your_key (optional)

    - ENVIRONMENT=development|production

    -REDIS_URL=redis://... (if using cache service)

---------------------    

### Testing

Run tests

    - pytest -q

Recommended test focus for the demo

    - Unit: indicators correctness, leakage guard, fetchers (with mocks), regression and classification wrappers.

    - Integration: data → features → model → predictions pipeline on small fixture.

    - Performance: quick smoke tests for training/pred speed.

---------------------    

### Roadmap (v2 Ideas)

- More data sources (Polygon/Quandl/FRED) and fundamentals integration.

- Advanced CV: Purged/K-fold variants, combinatorial purged CV.

- Portfolio management and risk parity allocation.

- Advanced ensembling and transformer-based models.

- Monitoring dashboards (Grafana/Prometheus) and alerting.

- Cloud deploy (Kubernetes/Terraform) and CI/CD pipelines.

- Authentication and RBAC for API/UI.

--------------------------------------

### License
This project is licensed under the MIT License. See LICENSE for details.

--------------

### Acknowledgments
- Streamlit, scikit-learn, Plotly, yfinance and the open-source community.
- Teammates and reviewers for rapid shipping under tight timelines.