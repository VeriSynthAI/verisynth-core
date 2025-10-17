# tests/test_synth.py
import pandas as pd
from verisynth.synth import generate_synthetic
from verisynth.schema import SchemaConfig

def test_generate_synthetic_shape(tmp_path):
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8],
        "smoker": [0, 1, 1]
    })
    synth_df, meta = generate_synthetic(df, n_rows=100, seed=42)
    assert len(synth_df) == 100
    assert set(df.columns) == set(synth_df.columns)
    assert meta["engine"].startswith("sdv")

def test_generate_synthetic_with_schema():
    """Test synthesis with schema configuration."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8],
        "smoker": [0, 1, 1]
    })
    
    # Create schema that excludes 'id' field
    schema_config = SchemaConfig(config_dict={
        'exclude': ['id'],
        'types': {'age': 'int', 'bmi': 'float', 'smoker': 'bool'}
    })
    
    synth_df, meta = generate_synthetic(df, n_rows=50, seed=42, schema_config=schema_config)
    
    # Verify schema was applied
    assert 'id' not in synth_df.columns
    assert len(synth_df) == 50
    assert meta['schema_applied'] is True
    assert meta['schema_config'] is not None

def test_generate_synthetic_without_schema():
    """Test synthesis without schema (backward compatibility)."""
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8],
        "smoker": [0, 1, 1]
    })
    
    synth_df, meta = generate_synthetic(df, n_rows=50, seed=42)
    
    # Verify no schema was applied
    assert meta['schema_applied'] is False
    assert meta['schema_config'] is None
    assert len(synth_df) == 50
