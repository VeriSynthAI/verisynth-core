<img src="https://github.com/user-attachments/assets/7e845625-51e1-4839-ac5a-3a34c8115d4f" alt="VeriSynth Logo" width="280"/>

# VeriSynth Core

[![Tests](https://github.com/VeriSynthAI/verisynth-core/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/VeriSynthAI/verisynth-core/actions/workflows/test.yml)


**VeriSynth Core** is a lightweight, privacy-preserving **synthetic data generation CLI**.
It transforms sensitive tabular datasets into statistically realistic synthetic data — with **cryptographic proof receipts** that verify integrity and reproducibility.

---

## ✨ Features

* 🔐 **Privacy-safe synthesis** — no real records are ever exposed
* 📊 **Statistical realism** using Gaussian Copula modeling
* 🗂️ **Schema configuration** — explicit field mapping and exclusion via YAML
* 🧾 **Proof receipts** (`proof.json`) include hashes, Merkle roots, metrics & seed
* 🧠 **Deterministic generation** via reproducible random seeds
* ⚡ **Runs locally** — no cloud or GPUs required
* 🧩 **Extensible** — drop-in engine architecture for future models (CTGAN, TVAE, etc.)

---

## 🧰 Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run VeriSynth

```bash
python -m verisynth.cli --input data/sample_patients.csv --output out/ --rows 1000
```

This command:

* Loads `data/sample_patients.csv`
* Learns its structure and correlations
* Generates 1,000 synthetic rows
* Saves:

  * `out/synthetic.csv` → synthetic dataset
  * `out/proof.json` → verifiable proof receipt

### 4. Optional: Use Schema Configuration

```bash
# Create example schema configuration
python -m verisynth.cli --create-schema-example config.yaml

# Run with schema (excludes patient_id, maps types)
python -m verisynth.cli --input data/sample_patients.csv --output out/ --schema config.yaml
```

---

## 🧪 Running Tests

VeriSynth includes a comprehensive test suite to ensure reliability and correctness.

### Prerequisites

Make sure you have the development dependencies installed:

```bash
pip install pytest pytest-cov
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=verisynth --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_cli.py -v

# Run tests in verbose mode with coverage
python -m pytest tests/ -v --cov=verisynth --cov-report=term-missing
```

### Test Structure

The test suite includes:

- **`test_cli.py`** - Tests the command-line interface functionality
- **`test_proof.py`** - Tests Merkle root consistency and proof generation
- **`test_synth.py`** - Tests synthetic data generation and shape validation
- **`test_schema.py`** - Tests schema configuration functionality (25 comprehensive test cases)

#### Schema Test Coverage

The `test_schema.py` file provides comprehensive testing for the schema feature:

- **Configuration Tests**: YAML file loading, validation, error handling
- **Field Operations**: Exclusion, type conversion (int, float, bool, str)
- **CLI Integration**: Schema file creation, command-line usage
- **Synthesis Integration**: Schema application during data generation
- **Edge Cases**: Empty dataframes, invalid configurations, missing fields
- **Backward Compatibility**: Ensures existing functionality still works

### Continuous Integration

Tests are automatically run on every push and pull request via GitHub Actions, ensuring code quality and preventing regressions.

---

## 🧾 Example Proof Receipt

```json
{
  "verisynth_version": "core-0.1.0",
  "license": "MIT",
  "metrics": { "corr_mean_abs_delta": 0.12, "naive_reid_risk": 0.01 },
  "input":  { "rows": 10, "sha256": "…82b7" },
  "output": { "rows": 1000000, "sha256": "…acb9" },
  "model":  { "engine": "GaussianCopula", "seed": 42 },
  "proof":  "merkle_root: …c31"
}
```

Each proof ensures **integrity** and **reproducibility**:
same input + same seed = identical output and Merkle proof.

## Verify Sample Proof
```bash
python verisynth/verify.py
```

---

## ⚙️ CLI Reference

```bash
python -m verisynth.cli --input <path> --output <dir> [--rows N] [--seed SEED] [--schema SCHEMA]
```

| Flag       | Description                                                   |
| ---------- | ------------------------------------------------------------- |
| `--input`  | Path to input CSV file                                        |
| `--output` | Output directory for synthetic data and proof                 |
| `--rows`   | Number of synthetic rows to generate (default: 1000)         |
| `--seed`   | Random seed for deterministic reproducibility                 |
| `--schema` | Path to YAML schema configuration file (optional)            |

Examples:

```bash
# Basic synthesis
python -m verisynth.cli --input data/finance.csv --output out/ --rows 500000 --seed 1337

# With schema configuration
python -m verisynth.cli --input data/patients.csv --output out/ --schema config.yaml

# Create example schema configuration
python -m verisynth.cli --create-schema-example config.yaml
```

---

## 🗂️ Schema Configuration

VeriSynth supports explicit field mapping and exclusion through YAML schema configuration files. This gives you fine-grained control over which fields to synthesize and how to handle data types.

### Schema Configuration Format

```yaml
exclude: ["patient_id", "address"]
types:
  age: int
  bmi: float
  smoker: bool
  hba1c: float
model:
  engine: GaussianCopula
  seed: 42
```

### Configuration Options

- **`exclude`**: List of field names to exclude from synthesis (e.g., IDs, addresses)
- **`types`**: Explicit type mappings for fields (supports: `int`, `float`, `bool`, `str`)
- **`model`**: Model configuration including engine and seed

### Benefits

- **Privacy**: Exclude sensitive identifiers and PII
- **Control**: Explicit type handling instead of automatic detection
- **Reproducibility**: Schema configuration is included in proof receipts
- **Validation**: Built-in validation ensures configuration correctness

---

## 🔬 What's Under the Hood

VeriSynth uses a **Gaussian Copula model** to learn the joint distribution of all numeric and categorical variables.
Instead of randomizing data, it captures real-world correlations (e.g., *age ↔ blood pressure*) and samples new records consistent with the original dataset.

---

## 🔒 Proof System

Each run produces a verifiable audit trail:

* **SHA-256 file hashes** of input/output
* **Merkle roots** linking dataset lineage
* **Model seed & parameters** for deterministic replay
* **Privacy & fidelity metrics**

> 🧩 This system provides verifiable lineage and reproducibility — a foundation for future zero-knowledge (ZK) verification.

---

## 🧠 Roadmap

* [ ] Add differential privacy metrics (ε, δ)
* [ ] Add support for CTGAN / TVAE models
* [ ] Add signed receipts (Ed25519)
* [ ] Add proof viewer

---

## 📜 License

MIT © [VeriSynth.ai](https://verisynth.ai)

---
