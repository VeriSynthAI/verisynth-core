# tests/test_cli.py
import subprocess
import os
import sys
import yaml

def test_cli_runs(tmp_path):
    input_csv = tmp_path / "data.csv"
    input_csv.write_text("age,bmi,smoker\n30,25.4,0\n40,29.1,1\n")

    # Use the same Python executable that's running the tests
    python_executable = sys.executable

    result = subprocess.run(
        [python_executable, "-m", "verisynth.cli",
         "--input", str(input_csv),
         "--output", str(tmp_path),
         "--rows", "10"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert (tmp_path / "synthetic.csv").exists()
    assert (tmp_path / "proof.json").exists()
    assert "VeriSynth" in result.stdout

def test_cli_with_schema(tmp_path):
    """Test CLI with schema configuration."""
    input_csv = tmp_path / "data.csv"
    input_csv.write_text("id,age,bmi,smoker\n1,30,25.4,0\n2,40,29.1,1\n")
    
    # Create schema config
    schema_file = tmp_path / "schema.yaml"
    schema_config = {
        'exclude': ['id'],
        'types': {'age': 'int', 'bmi': 'float', 'smoker': 'bool'}
    }
    with open(schema_file, 'w') as f:
        yaml.dump(schema_config, f)
    
    python_executable = sys.executable
    result = subprocess.run(
        [python_executable, "-m", "verisynth.cli",
         "--input", str(input_csv),
         "--output", str(tmp_path / "output"),
         "--schema", str(schema_file),
         "--rows", "10"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Loaded schema configuration" in result.stdout
    assert "Applied schema configuration" in result.stdout
    
    # Check output files
    output_dir = tmp_path / "output"
    assert (output_dir / "synthetic.csv").exists()
    assert (output_dir / "proof.json").exists()
    
    # Verify schema was applied (ID should be excluded)
    import pandas as pd
    synth_df = pd.read_csv(output_dir / "synthetic.csv")
    assert 'id' not in synth_df.columns
    assert len(synth_df) == 10

def test_cli_create_schema_example(tmp_path):
    """Test CLI schema example creation."""
    example_file = tmp_path / "example.yaml"
    
    python_executable = sys.executable
    result = subprocess.run(
        [python_executable, "-m", "verisynth.cli",
         "--create-schema-example", str(example_file)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert example_file.exists()
    assert "Example schema configuration created" in result.stdout
