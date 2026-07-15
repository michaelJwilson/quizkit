# quizkit
A Python framework for basic quantum computing.

## Prerequisites
* **Python:** `>=3.12.2`
* **Package Manager:** [`uv`](https://github.com/astral-sh/uv)

## Installation
### 1. Create a Virtual Environment
Use `uv` to create a fresh virtual environment specifying the required Python version:

```bash
# Create the virtual environment
uv venv --python 3.12

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

## Package & core dependency install
uv pip install -e .

## Dev install
uv pip install -e ".[dev,test,build]"

## Usage
run_quizkit

## package layout
├── pyproject.toml
├── README.md
├── quizkit.log
├── python/
│   └── quizkit/
│       ├── __init__.py
│       └── plots/
│       └── results/
│           └── results.parquet
│       └── scripts/
│           └── run_quizkit.py
│       └── data/
│           └── input.csv
│           └──	input2.json
