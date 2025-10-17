\
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, Any, Optional
from .schema import SchemaConfig

# Suppress SDV deprecation warnings globally
warnings.filterwarnings("ignore", category=FutureWarning, module="sdv")
warnings.filterwarnings("ignore", category=UserWarning, module="sdv")

def _try_import_sdv():
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        return GaussianCopulaSynthesizer, SingleTableMetadata
    except Exception as e:
        return None, None

def fit_and_sample(df: pd.DataFrame, n_rows: int, seed: int = 42, schema_config: Optional[SchemaConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tries SDV GaussianCopula first. If unavailable, falls back to per-column sampling.
    
    Args:
        df: Input DataFrame
        n_rows: Number of rows to generate
        seed: Random seed
        schema_config: Optional schema configuration for field mapping and exclusion
    """
    np.random.seed(seed)

    # Apply schema configuration if provided
    if schema_config:
        # Validate schema configuration
        validation_errors = schema_config.validate()
        if validation_errors:
            raise ValueError(f"Schema configuration validation failed: {'; '.join(validation_errors)}")
        
        # Apply schema to input DataFrame
        df = schema_config.apply_to_dataframe(df)
        print(f"Applied schema configuration: excluded {len(schema_config.exclude_fields)} fields, mapped {len(schema_config.field_types)} field types")

    GaussianCopulaSynthesizer, SingleTableMetadata = _try_import_sdv()
    meta = {}

    if GaussianCopulaSynthesizer is not None:
        # Build metadata automatically - suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)

        model = GaussianCopulaSynthesizer(metadata)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)

        #synthetic = model.sample(num_rows=n_rows, randomize_samples=False)
        synthetic = model.sample(num_rows=n_rows)
        
        meta = {
            "engine": "sdv.GaussianCopulaSynthesizer",
            "detected_dtypes": df.dtypes.astype(str).to_dict(),
            "schema_applied": schema_config is not None,
            "schema_config": schema_config.to_dict() if schema_config else None,
        }
        return synthetic, meta

    # Fallback sampler (independent columns) – day-1 only
    synthetic = pd.DataFrame()

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            # sample with replacement from empirical values (preserve spikes)
            synthetic[col] = np.random.choice(series.dropna().values, size=n_rows, replace=True)
        else:
            # categorical sampling by frequency
            vals = series.dropna().values
            if len(vals) == 0:
                synthetic[col] = pd.Series([None]*n_rows)
            else:
                synthetic[col] = np.random.choice(vals, size=n_rows, replace=True)

    meta = {
        "engine": "fallback.empirical_sampler", 
        "detected_dtypes": df.dtypes.astype(str).to_dict(),
        "schema_applied": schema_config is not None,
        "schema_config": schema_config.to_dict() if schema_config else None,
    }
    return synthetic, meta

def generate_synthetic(df: pd.DataFrame, n_rows: int, seed: int = 42, schema_config: Optional[SchemaConfig] = None):
    """
    Generate synthetic data with optional schema configuration.
    
    Args:
        df: Input DataFrame
        n_rows: Number of rows to generate
        seed: Random seed
        schema_config: Optional schema configuration for field mapping and exclusion
    """
    return fit_and_sample(df, n_rows=n_rows, seed=seed, schema_config=schema_config)
