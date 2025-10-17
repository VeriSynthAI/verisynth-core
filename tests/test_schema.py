# tests/test_schema.py
import pytest
import pandas as pd
import yaml
import tempfile
import os
from pathlib import Path
from verisynth.schema import SchemaConfig, create_example_config


class TestSchemaConfig:
    """Test SchemaConfig class functionality."""
    
    def test_init_with_dict(self):
        """Test initialization with dictionary."""
        config_dict = {
            'exclude': ['id', 'name'],
            'types': {'age': 'int', 'score': 'float'},
            'model': {'engine': 'GaussianCopula', 'seed': 42}
        }
        schema = SchemaConfig(config_dict=config_dict)
        
        assert schema.exclude_fields == ['id', 'name']
        assert schema.field_types == {'age': 'int', 'score': 'float'}
        assert schema.get_engine() == 'GaussianCopula'
        assert schema.get_seed() == 42
    
    def test_init_with_file(self, tmp_path):
        """Test initialization with YAML file."""
        config_dict = {
            'exclude': ['patient_id'],
            'types': {'age': 'int', 'bmi': 'float'},
            'model': {'engine': 'GaussianCopula', 'seed': 123}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        schema = SchemaConfig(config_path=str(config_file))
        
        assert schema.exclude_fields == ['patient_id']
        assert schema.field_types == {'age': 'int', 'bmi': 'float'}
        assert schema.get_seed() == 123
    
    def test_init_empty(self):
        """Test initialization with empty configuration."""
        schema = SchemaConfig()
        
        assert schema.exclude_fields == []
        assert schema.field_types == {}
        assert schema.get_engine() == 'GaussianCopula'
        assert schema.get_seed() is None
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            SchemaConfig(config_path="nonexistent.yaml")
    
    def test_invalid_yaml(self, tmp_path):
        """Test error handling for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            SchemaConfig(config_path=str(config_file))


class TestSchemaValidation:
    """Test schema configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'exclude': ['id', 'name'],
            'types': {'age': 'int', 'score': 'float'},
            'model': {'engine': 'GaussianCopula', 'seed': 42}
        }
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert errors == []
    
    def test_invalid_exclude_type(self):
        """Test validation with invalid exclude type."""
        config = {'exclude': 'not_a_list'}
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert any("must be a list" in error for error in errors)
    
    def test_invalid_types_type(self):
        """Test validation with invalid types type."""
        config = {'types': 'not_a_dict'}
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert any("must be a dictionary" in error for error in errors)
    
    def test_invalid_field_type(self):
        """Test validation with invalid field type."""
        config = {'types': {'age': 'invalid_type'}}
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert any("Unsupported type" in error for error in errors)
    
    def test_invalid_model_config(self):
        """Test validation with invalid model configuration."""
        config = {'model': 'not_a_dict'}
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert any("must be a dictionary" in error for error in errors)
    
    def test_invalid_seed_type(self):
        """Test validation with invalid seed type."""
        config = {'model': {'seed': 'not_an_int'}}
        schema = SchemaConfig(config_dict=config)
        errors = schema.validate()
        assert any("must be an integer" in error for error in errors)


class TestSchemaApplication:
    """Test schema application to DataFrames."""
    
    def test_exclude_fields(self):
        """Test field exclusion."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'age': [30, 40, 50],
            'score': [85.5, 92.1, 78.3]
        })
        
        config = {'exclude': ['id', 'name']}
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        assert 'id' not in result_df.columns
        assert 'name' not in result_df.columns
        assert 'age' in result_df.columns
        assert 'score' in result_df.columns
        assert len(result_df) == 3
    
    def test_type_conversion(self):
        """Test type conversion."""
        df = pd.DataFrame({
            'age': ['30', '40', '50'],  # String numbers
            'score': ['85.5', '92.1', '78.3'],  # String floats
            'active': ['True', 'False', 'True'],  # String booleans
            'name': ['A', 'B', 'C']  # String text
        })
        
        config = {
            'types': {
                'age': 'int',
                'score': 'float',
                'active': 'bool',
                'name': 'str'
            }
        }
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        assert result_df['age'].dtype.name == 'Int64'
        assert result_df['score'].dtype.name == 'float64'
        assert result_df['active'].dtype.name == 'boolean'
        assert result_df['name'].dtype.name == 'string'
    
    def test_missing_fields_warning(self, capsys):
        """Test warning for missing fields."""
        df = pd.DataFrame({'age': [30, 40, 50]})
        
        config = {
            'exclude': ['nonexistent_field'],
            'types': {'nonexistent_field': 'int'}
        }
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        captured = capsys.readouterr()
        assert "not found in data" in captured.out
    
    def test_type_conversion_error_handling(self, capsys):
        """Test error handling during type conversion."""
        df = pd.DataFrame({'age': ['invalid', '40', '50']})
        
        config = {'types': {'age': 'int'}}
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        # Should handle conversion errors gracefully
        captured = capsys.readouterr()
        # The conversion should still work with NaN for invalid values
        assert result_df['age'].dtype.name == 'Int64'


class TestSchemaIntegration:
    """Test schema integration with synthesis."""
    
    def test_synthesis_with_schema(self):
        """Test synthesis with schema configuration."""
        from verisynth.synth import generate_synthetic
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'age': [30, 40, 50],
            'bmi': [24.5, 30.2, 27.8],
            'smoker': [0, 1, 1]
        })
        
        config = {
            'exclude': ['id'],
            'types': {'age': 'int', 'bmi': 'float', 'smoker': 'bool'}
        }
        schema = SchemaConfig(config_dict=config)
        
        synth_df, meta = generate_synthetic(df, n_rows=50, seed=42, schema_config=schema)
        
        # Check that ID was excluded
        assert 'id' not in synth_df.columns
        assert len(synth_df) == 50
        
        # Check metadata includes schema info
        assert meta['schema_applied'] is True
        assert meta['schema_config'] is not None
        assert meta['schema_config']['exclude'] == ['id']
    
    def test_synthesis_without_schema(self):
        """Test synthesis without schema (backward compatibility)."""
        from verisynth.synth import generate_synthetic
        
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'bmi': [24.5, 30.2, 27.8],
            'smoker': [0, 1, 1]
        })
        
        synth_df, meta = generate_synthetic(df, n_rows=50, seed=42)
        
        # Check metadata indicates no schema
        assert meta['schema_applied'] is False
        assert meta['schema_config'] is None


class TestSchemaCLI:
    """Test CLI integration with schema."""
    
    def test_create_schema_example(self, tmp_path):
        """Test creating schema example file."""
        example_file = tmp_path / "example.yaml"
        
        # This would normally be called via CLI, but we can test the function directly
        create_example_config(str(example_file))
        
        assert example_file.exists()
        
        # Verify the content
        with open(example_file, 'r') as f:
            content = yaml.safe_load(f)
        
        assert 'exclude' in content
        assert 'types' in content
        assert 'model' in content
        assert content['model']['engine'] == 'GaussianCopula'
    
    def test_cli_with_schema(self, tmp_path):
        """Test CLI with schema configuration."""
        import subprocess
        import sys
        
        # Create test data
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
        
        # Run CLI with schema
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
        synth_df = pd.read_csv(output_dir / "synthetic.csv")
        assert 'id' not in synth_df.columns
        assert len(synth_df) == 10
    
    def test_cli_create_schema_example(self, tmp_path):
        """Test CLI schema example creation."""
        import subprocess
        import sys
        
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
    
    def test_cli_invalid_schema(self, tmp_path):
        """Test CLI with invalid schema file."""
        import subprocess
        import sys
        
        # Create test data
        input_csv = tmp_path / "data.csv"
        input_csv.write_text("age,bmi\n30,25.4\n40,29.1\n")
        
        # Create invalid schema config
        schema_file = tmp_path / "invalid_schema.yaml"
        schema_file.write_text("invalid: yaml: content: [")
        
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, "-m", "verisynth.cli",
             "--input", str(input_csv),
             "--output", str(tmp_path / "output"),
             "--schema", str(schema_file)],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "Error loading schema configuration" in result.stdout


class TestSchemaEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Test schema application to empty DataFrame."""
        df = pd.DataFrame()
        
        config = {'exclude': ['id'], 'types': {'age': 'int'}}
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        assert len(result_df) == 0
    
    def test_all_fields_excluded(self):
        """Test excluding all fields."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        
        config = {'exclude': ['id', 'name']}
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        assert len(result_df.columns) == 0
        assert len(result_df) == 2
    
    def test_nonexistent_exclude_fields(self):
        """Test excluding fields that don't exist."""
        df = pd.DataFrame({'age': [30, 40]})
        
        config = {'exclude': ['nonexistent1', 'nonexistent2']}
        schema = SchemaConfig(config_dict=config)
        result_df = schema.apply_to_dataframe(df)
        
        # Should still work, just with warnings
        assert 'age' in result_df.columns
        assert len(result_df) == 2
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = {
            'exclude': ['id'],
            'types': {'age': 'int'},
            'model': {'engine': 'GaussianCopula', 'seed': 42}
        }
        schema = SchemaConfig(config_dict=config)
        result_dict = schema.to_dict()
        
        assert result_dict == config
        assert result_dict is not config  # Should be a copy
