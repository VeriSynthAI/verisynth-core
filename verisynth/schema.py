"""
Schema configuration module for VeriSynth Core.

Handles YAML-based schema configuration for explicit field mapping and exclusion.
"""

import yaml
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class SchemaConfig:
    """Schema configuration handler for VeriSynth."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize schema configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_path:
            self.config = self._load_from_file(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = {}
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in schema configuration: {e}")
    
    @property
    def exclude_fields(self) -> List[str]:
        """Get list of fields to exclude from synthesis."""
        return self.config.get('exclude', [])
    
    @property
    def field_types(self) -> Dict[str, str]:
        """Get explicit field type mappings."""
        return self.config.get('types', {})
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_engine(self) -> str:
        """Get the synthesis engine name."""
        return self.model_config.get('engine', 'GaussianCopula')
    
    def get_seed(self) -> Optional[int]:
        """Get the random seed."""
        return self.model_config.get('seed')
    
    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply schema configuration to a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with schema applied (excluded fields removed, types converted)
        """
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Exclude specified fields
        exclude_fields = self.exclude_fields
        if exclude_fields:
            missing_fields = [field for field in exclude_fields if field not in result_df.columns]
            if missing_fields:
                print(f"Warning: Fields specified for exclusion not found in data: {missing_fields}")
            
            # Remove excluded fields
            fields_to_keep = [col for col in result_df.columns if col not in exclude_fields]
            result_df = result_df[fields_to_keep]
        
        # Apply explicit type conversions
        field_types = self.field_types
        for field, target_type in field_types.items():
            if field not in result_df.columns:
                print(f"Warning: Field '{field}' specified in types but not found in data")
                continue
            
            try:
                if target_type.lower() == 'int':
                    result_df[field] = pd.to_numeric(result_df[field], errors='coerce').astype('Int64')
                elif target_type.lower() == 'float':
                    result_df[field] = pd.to_numeric(result_df[field], errors='coerce').astype('float64')
                elif target_type.lower() == 'bool':
                    result_df[field] = result_df[field].astype('boolean')
                elif target_type.lower() == 'str' or target_type.lower() == 'string':
                    result_df[field] = result_df[field].astype('string')
                else:
                    print(f"Warning: Unsupported type '{target_type}' for field '{field}', keeping original type")
            except Exception as e:
                print(f"Warning: Failed to convert field '{field}' to type '{target_type}': {e}")
        
        return result_df
    
    def validate(self) -> List[str]:
        """
        Validate the schema configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate exclude fields
        exclude_fields = self.exclude_fields
        if not isinstance(exclude_fields, list):
            errors.append("'exclude' must be a list of field names")
        elif exclude_fields:
            for field in exclude_fields:
                if not isinstance(field, str):
                    errors.append(f"Exclude field '{field}' must be a string")
        
        # Validate field types
        field_types = self.field_types
        if not isinstance(field_types, dict):
            errors.append("'types' must be a dictionary mapping field names to types")
        else:
            valid_types = {'int', 'float', 'bool', 'str', 'string'}
            for field, field_type in field_types.items():
                if not isinstance(field, str):
                    errors.append(f"Field name '{field}' must be a string")
                elif not isinstance(field_type, str):
                    errors.append(f"Type for field '{field}' must be a string")
                elif field_type.lower() not in valid_types:
                    errors.append(f"Unsupported type '{field_type}' for field '{field}'. Supported types: {valid_types}")
        
        # Validate model configuration
        model_config = self.model_config
        if not isinstance(model_config, dict):
            errors.append("'model' must be a dictionary")
        else:
            engine = model_config.get('engine')
            if engine and not isinstance(engine, str):
                errors.append("Model 'engine' must be a string")
            
            seed = model_config.get('seed')
            if seed is not None and not isinstance(seed, int):
                errors.append("Model 'seed' must be an integer")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()


def create_example_config(output_path: str) -> None:
    """
    Create an example schema configuration file.
    
    Args:
        output_path: Path where to save the example configuration
    """
    example_config = {
        'exclude': ['patient_id', 'address'],
        'types': {
            'age': 'int',
            'bmi': 'float',
            'smoker': 'bool',
            'hba1c': 'float'
        },
        'model': {
            'engine': 'GaussianCopula',
            'seed': 42
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Example schema configuration created at: {output_path}")
