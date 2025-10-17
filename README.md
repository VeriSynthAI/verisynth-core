![logo](https://github.com/user-attachments/assets/7e845625-51e1-4839-ac5a-3a34c8115d4f)

# VeriSynth Core

**VeriSynth Core** is a lightweight, privacy-preserving **synthetic data generation CLI**.
It transforms sensitive tabular datasets into statistically realistic synthetic data — with **cryptographic proof receipts** that verify integrity and reproducibility.

---

## ✨ Features

* 🔐 **Privacy-safe synthesis** — no real records are ever exposed
* 📊 **Statistical realism** using Gaussian Copula modeling
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
verisynth synth --input <path> --output <dir> [--rows N] [--seed SEED]
```

| Flag       | Description                                                   |
| ---------- | ------------------------------------------------------------- |
| `--input`  | Path to input CSV file                                        |
| `--output` | Output directory for synthetic data and proof                 |
| `--rows`   | Number of synthetic rows to generate (default: same as input) |
| `--seed`   | Random seed for deterministic reproducibility                 |

Example:

```bash
verisynth synth data/finance.csv -o out/ --rows 500000 --seed 1337
```

---

## 🔬 What’s Under the Hood

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
