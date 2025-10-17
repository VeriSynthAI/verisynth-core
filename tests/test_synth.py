# tests/test_synth.py
import pytest
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

def test_generate_synthetic_fallback_mode():
    """Test synthesis in fallback mode (when SDV is not available)."""
    from verisynth.synth import fit_and_sample
    
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8],
        "smoker": [0, 1, 1]
    })
    
    # Mock SDV import failure
    import unittest.mock
    with unittest.mock.patch('verisynth.synth._try_import_sdv', return_value=(None, None)):
        synth_df, meta = fit_and_sample(df, n_rows=50, seed=42)
        
        assert len(synth_df) == 50
        assert meta['engine'] == 'fallback.empirical_sampler'
        assert 'age' in synth_df.columns
        assert 'bmi' in synth_df.columns
        assert 'smoker' in synth_df.columns

def test_generate_synthetic_with_invalid_schema():
    """Test synthesis with invalid schema configuration."""
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8]
    })
    
    # Create invalid schema config
    invalid_schema = SchemaConfig(config_dict={
        'exclude': 'not_a_list',  # Invalid type
        'types': {'age': 'invalid_type'}  # Invalid type
    })
    
    with pytest.raises(ValueError, match="Schema configuration validation failed"):
        generate_synthetic(df, n_rows=50, seed=42, schema_config=invalid_schema)

def test_generate_synthetic_schema_validation_error():
    """Test synthesis with schema validation errors."""
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [24.5, 30.2, 27.8]
    })
    
    # Create schema with validation errors
    schema_with_errors = SchemaConfig(config_dict={
        'exclude': ['nonexistent_field'],
        'types': {'nonexistent_field': 'int'}
    })
    
    # Should not raise error, but should handle gracefully
    synth_df, meta = generate_synthetic(df, n_rows=50, seed=42, schema_config=schema_with_errors)
    
    assert len(synth_df) == 50
    assert meta['schema_applied'] is True
