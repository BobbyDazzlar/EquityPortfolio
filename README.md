# Equity Portfolio Optimization

Research notebooks, scripts, datasets, and experiment outputs for building optimized Indian equity portfolios. The project compares classical Mean-Variance Optimization (MVO) with metaheuristic approaches such as Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO), using Sharpe and Sortino ratios as objective functions.

## Project Overview

This repository was developed as a final-year research project around portfolio construction for the Indian equity market. It works with NIFTY 50 and NIFTY 500 stock data, applies filtering and forecasting experiments, then evaluates optimized portfolios over static and dynamic time windows.

The core workflow is:

1. Prepare NIFTY stock and index datasets.
2. Filter candidate equities using historical performance and LSTM-based experiments.
3. Generate portfolio weights with MVO, PSO, and ACO.
4. Evaluate portfolios with risk-adjusted metrics such as Sharpe and Sortino ratios.
5. Compare results against NIFTY 50 benchmark outputs.

## Methods Included

| Method | Purpose | Main outputs |
| --- | --- | --- |
| MVO | Monte Carlo-style mean-variance portfolio generation and optimal weight selection | Optimized Sharpe/Sortino portfolios and weight summaries |
| PSO | Particle Swarm Optimization for portfolio weight search | Static and dynamic weight windows, tuned Sharpe/Sortino portfolios |
| ACO | Ant Colony Optimization with Optuna-based hyperparameter tuning | ACO portfolio weights and hyperparameter search outputs |
| LSTM | Stock filtering and prediction experiments | Filtered stock universes for optimization experiments |
| Benchmarking | NIFTY 50 comparison baseline | Index-level return and summary CSV files |

## Repository Structure

```text
.
|-- phase2_final results/
|   |-- ACO weights/                 # Final ACO portfolio weights and summaries
|   |-- MVO weights/                 # Final MVO portfolio weights and summaries
|   |-- Nifty50_benchmark/           # Benchmark notebooks and CSV summaries
|   |-- PSO weigths/                 # Final PSO portfolio weights and summaries
|   `-- Static weight window/        # Static-window MVO, PSO, and ACO experiments
|-- OneDrive_2021-09-24/
|   `-- Final Year Project Drive/
|       |-- Dataset/                 # NIFTY 50/NIFTY 500 datasets and factsheets
|       |-- codes/                   # Earlier experiment notebooks
|       `-- codes_2022/              # Main static/dynamic experiment scripts
|-- Literature_survey/               # Reference papers and survey material
|-- college Review doc/              # Review decks and academic presentation files
|-- LICENSE
`-- README.md
```

Note: some directory names preserve the original project archive spelling, including `PSO weigths`.

## Key Files

- `phase2_final results/Results comparison.xlsx` - consolidated comparison workbook for the final experiments.
- `phase2_final results/Nifty50_benchmark/Nifty50.ipynb` - benchmark generation notebook.
- `phase2_final results/MVO weights/2021_MVO_weights_summary.ipynb` - final MVO weight summary notebook.
- `phase2_final results/PSO weigths/2021_PSO_weights_summary.ipynb` - final PSO weight summary notebook.
- `phase2_final results/ACO weights/2021_ACO_weights_summary.ipynb` - final ACO weight summary notebook.
- `phase2_final results/Static weight window/ACO/Static_ACO.py` - static-window ACO implementation with Optuna tuning.
- `OneDrive_2021-09-24/Final Year Project Drive/Dataset/` - raw and filtered market data used by the notebooks and scripts.

## Tech Stack

The experiments are primarily Python and Jupyter based.

- Python
- Jupyter Notebook
- pandas
- NumPy
- matplotlib
- scikit-learn
- TensorFlow / Keras
- Optuna

## Getting Started

Clone the repository:

```bash
git clone https://github.com/BobbyDazzlar/EquityPortfolio.git
cd EquityPortfolio
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main analysis dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Start with the notebooks under `phase2_final results/` if you want the final experiment outputs, or explore `OneDrive_2021-09-24/Final Year Project Drive/codes_2022/` for the broader experiment pipeline.

## Reproducibility Notes

This repository is an academic research archive, so several notebooks and scripts preserve absolute paths from the original machine or cloud notebook environment. Before rerunning an experiment, update dataset paths to match your local checkout.

For example, scripts may reference paths like:

```text
/home/pn_kumar/...
/content/drive/My Drive/...
```

Replace those with paths inside this repository, especially under:

```text
OneDrive_2021-09-24/Final Year Project Drive/Dataset/
phase2_final results/
```

Some notebooks include saved outputs and checkpoint artifacts. For new experiments, prefer writing fresh outputs into a separate results directory so previous academic results remain traceable.

## Results

The final comparison files are stored in:

- `phase2_final results/Results comparison.xlsx`
- `phase2_final results/Results comparison_viz.xlsx`
- `phase2_final results/Nifty50_benchmark/`
- `phase2_final results/ACO weights/`
- `phase2_final results/MVO weights/`
- `phase2_final results/PSO weigths/`

These files compare optimized portfolio performance across MVO, PSO, and ACO strategies, including Sharpe-optimized and Sortino-optimized variants.

## Suggested Cleanup Roadmap

The repository currently keeps the full research archive in one place. A future cleanup could make it easier to maintain by:

- Moving reusable code into a `src/` package.
- Moving raw data and generated outputs into clearly separated `data/` and `results/` folders.
- Removing tracked `.ipynb_checkpoints` and IDE metadata from version control.
- Pinning historical dependency versions in `requirements.txt` or adding an `environment.yml`.
- Replacing absolute paths in notebooks and scripts with configuration-driven relative paths.
- Converting final notebooks into reproducible scripts or documented pipelines.

## Disclaimer

This repository is for academic and educational research. It is not financial advice, and the results should not be used as investment recommendations without independent validation.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
