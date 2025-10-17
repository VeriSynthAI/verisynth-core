\
import argparse, os, json, time, random, platform, sys
import pandas as pd
import numpy as np

from .synth import generate_synthetic
from .privacy import summarize_privacy_fidelity
from .lineage import dataframe_merkle, file_hash
from .schema import SchemaConfig

def main():
    ap = argparse.ArgumentParser(description="VeriSynth Micro — local synthetic generator with proof receipt")
    ap.add_argument("--input", help="Path to input CSV")
    ap.add_argument("--output", help="Output directory")
    ap.add_argument("--rows", type=int, default=1000, help="Rows to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--schema", help="Path to YAML schema configuration file")
    ap.add_argument("--create-schema-example", help="Create example schema configuration file at specified path")
    args = ap.parse_args()

    # Handle schema example creation
    if args.create_schema_example:
        from .schema import create_example_config
        create_example_config(args.create_schema_example)
        return

    # Validate required arguments for synthesis
    if not args.input or not args.output:
        ap.error("--input and --output are required for synthesis")

    os.makedirs(args.output, exist_ok=True)

    # Load schema configuration if provided
    schema_config = None
    if args.schema:
        try:
            schema_config = SchemaConfig(config_path=args.schema)
            print(f"Loaded schema configuration from: {args.schema}")
        except Exception as e:
            print(f"Error loading schema configuration: {e}")
            sys.exit(1)

    # Load
    df = pd.read_csv(args.input)
    model_seed = args.seed
    np.random.seed(model_seed)
    random.seed(model_seed)

    # Synthesize
    synth, meta = generate_synthetic(df, n_rows=args.rows, seed=model_seed, schema_config=schema_config)

    # Metrics
    metrics = summarize_privacy_fidelity(df, synth)

    # Lineage roots
    in_merkle = dataframe_merkle(df)
    out_merkle = dataframe_merkle(synth)

    # Write outputs
    synth_path = os.path.join(args.output, "synthetic.csv")
    proof_path = os.path.join(args.output, "proof.json")
    report_path = os.path.join(args.output, "report.txt")

    synth.to_csv(synth_path, index=False)

    proof = {
        "verisynth_version": "micro-0.1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "path": os.path.abspath(args.input),
            "file_sha256": file_hash(args.input),
            "merkle_root": in_merkle["merkle_root"],
            "row_count": int(in_merkle["row_hashes_count"]),
        },
        "output": {
            "path": os.path.abspath(synth_path),
            "file_sha256": file_hash(synth_path),
            "merkle_root": out_merkle["merkle_root"],
            "row_count": int(out_merkle["row_hashes_count"]),
        },
        "model": {
            "engine": meta.get("engine", "unknown"),
            "seed": model_seed,
            "detected_dtypes": meta.get("detected_dtypes", {}),
            "schema_applied": meta.get("schema_applied", False),
            "schema_config": meta.get("schema_config"),
        },
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "disclaimer": "Prototype only. Not a formal privacy guarantee."
    }
    with open(proof_path, "w") as f:
        json.dump(proof, f, indent=2)

    # Human-readable report
    lines = []
    lines.append("VeriSynth Micro — Synthesis Report")
    lines.append("="*40)
    lines.append(f"Input file: {args.input}")
    lines.append(f"Output file: {synth_path}")
    lines.append(f"Engine: {proof['model']['engine']}  | Seed: {model_seed}")
    if proof['model']['schema_applied']:
        lines.append(f"Schema: Applied from {args.schema}")
        schema_config = proof['model']['schema_config']
        if schema_config:
            if schema_config.get('exclude'):
                lines.append(f"  Excluded fields: {', '.join(schema_config['exclude'])}")
            if schema_config.get('types'):
                lines.append(f"  Type mappings: {len(schema_config['types'])} fields")
    lines.append("")
    lines.append("Fidelity & Privacy Diagnostics:")
    lines.append(f"- Mean Abs Correlation Delta: {metrics['corr_mean_abs_delta']} (lower is better)")
    lines.append(f"- Naive Re-ID Risk Fraction: {metrics['naive_reid_risk_fraction']} (lower is better)")
    lines.append("- KS p-values by column (higher ~ more similar):")
    for k, v in metrics["ks_pvalues"].items():
        lines.append(f"    {k}: p={v:.4f}")
    lines.append("")
    lines.append("Lineage (Merkle roots):")
    lines.append(f"- Input  Merkle: {in_merkle['merkle_root']}")
    lines.append(f"- Output Merkle: {out_merkle['merkle_root']}")
    lines.append("")
    lines.append("Proof JSON written to proof.json (shareable receipt)")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

if __name__ == "__main__":
    main()
