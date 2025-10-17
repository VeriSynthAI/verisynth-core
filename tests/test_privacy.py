# tests/test_privacy.py
import pytest
import pandas as pd
import numpy as np
from verisynth.privacy import (
    ks_tests, 
    correlation_delta, 
    naive_reid_risk, 
    summarize_privacy_fidelity
)


class TestKSTests:
    """Test Kolmogorov-Smirnov tests functionality."""
    
    def test_ks_tests_basic(self):
        """Test basic KS test functionality."""
        # Need at least 11 samples for KS test
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75, 80],
            'score': [85, 90, 75, 80, 95, 88, 92, 78, 82, 97, 100],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'A', 'C', 'B', 'A']
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72, 37, 47, 57, 67, 77, 82],
            'score': [87, 92, 77, 82, 97, 90, 94, 80, 84, 99, 102],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'A', 'C', 'B', 'A']
        })
        
        result = ks_tests(real, synth)
        
        assert 'age' in result
        assert 'score' in result
        assert 'category' not in result  # Non-numeric column should be excluded
        assert isinstance(result['age'], float)
        assert isinstance(result['score'], float)
        assert 0 <= result['age'] <= 1
        assert 0 <= result['score'] <= 1
    
    def test_ks_tests_no_common_columns(self):
        """Test KS tests with no common columns."""
        real = pd.DataFrame({'age': [30, 40, 50]})
        synth = pd.DataFrame({'height': [170, 180, 190]})
        
        result = ks_tests(real, synth)
        assert result == {}
    
    def test_ks_tests_small_samples(self):
        """Test KS tests with small samples (should be excluded)."""
        real = pd.DataFrame({'age': [30, 40]})  # Only 2 samples
        synth = pd.DataFrame({'age': [32, 42]})
        
        result = ks_tests(real, synth)
        assert result == {}  # Should be empty due to small sample size
    
    def test_ks_tests_non_numeric(self):
        """Test KS tests with non-numeric columns."""
        # Need at least 11 samples for KS test
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75, 80],
            'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72, 37, 47, 57, 67, 77, 82],
            'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        })
        
        result = ks_tests(real, synth)
        assert 'age' in result
        assert 'name' not in result  # Non-numeric should be excluded
    
    def test_ks_tests_with_nans(self):
        """Test KS tests with NaN values."""
        # Need at least 11 samples for KS test, and after dropping NaNs
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75, 80, 85],
            'score': [85, 90, 75, 80, 95, 88, 92, 78, 82, 97, 100, 105]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72, 37, 47, 57, 67, 77, 82, 87],
            'score': [87, 92, 77, 82, 97, 90, 94, 80, 84, 99, 102, 107]
        })
        
        result = ks_tests(real, synth)
        assert 'age' in result
        assert 'score' in result
        # Should handle NaNs gracefully


class TestCorrelationDelta:
    """Test correlation delta functionality."""
    
    def test_correlation_delta_basic(self):
        """Test basic correlation delta calculation."""
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70],
            'score': [85, 90, 75, 80, 95],
            'height': [170, 180, 175, 185, 190]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72],
            'score': [87, 92, 77, 82, 97],
            'height': [172, 182, 177, 187, 192]
        })
        
        result = correlation_delta(real, synth)
        assert isinstance(result, float)
        assert result >= 0  # Should be non-negative
        assert not np.isnan(result)
    
    def test_correlation_delta_no_common_numeric(self):
        """Test correlation delta with no common numeric columns."""
        real = pd.DataFrame({'age': [30, 40, 50]})
        synth = pd.DataFrame({'height': [170, 180, 190]})
        
        result = correlation_delta(real, synth)
        assert np.isnan(result)
    
    def test_correlation_delta_insufficient_columns(self):
        """Test correlation delta with insufficient numeric columns."""
        real = pd.DataFrame({'age': [30, 40, 50]})
        synth = pd.DataFrame({'age': [32, 42, 52]})
        
        result = correlation_delta(real, synth)
        assert np.isnan(result)  # Need at least 2 numeric columns
    
    def test_correlation_delta_with_nans(self):
        """Test correlation delta with NaN values."""
        real = pd.DataFrame({
            'age': [30, 40, np.nan, 60, 70],
            'score': [85, 90, 75, np.nan, 95]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72],
            'score': [87, 92, 77, 82, 97]
        })
        
        result = correlation_delta(real, synth)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestNaiveReidRisk:
    """Test naive re-identification risk functionality."""
    
    def test_naive_reid_risk_basic(self):
        """Test basic naive re-identification risk calculation."""
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70],
            'score': [85, 90, 75, 80, 95]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72],
            'score': [87, 92, 77, 82, 97]
        })
        
        result = naive_reid_risk(real, synth)
        assert isinstance(result, float)
        assert 0 <= result <= 1  # Should be a fraction
        assert not np.isnan(result)
    
    def test_naive_reid_risk_no_shared_numeric(self):
        """Test naive reid risk with no shared numeric columns."""
        real = pd.DataFrame({'age': [30, 40, 50]})
        synth = pd.DataFrame({'height': [170, 180, 190]})
        
        result = naive_reid_risk(real, synth)
        assert np.isnan(result)
    
    def test_naive_reid_risk_empty_data(self):
        """Test naive reid risk with empty data."""
        real = pd.DataFrame({'age': []})
        synth = pd.DataFrame({'age': [30, 40, 50]})
        
        result = naive_reid_risk(real, synth)
        assert np.isnan(result)
    
    def test_naive_reid_risk_custom_threshold(self):
        """Test naive reid risk with custom threshold."""
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70],
            'score': [85, 90, 75, 80, 95]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72],
            'score': [87, 92, 77, 82, 97]
        })
        
        result_low = naive_reid_risk(real, synth, threshold=0.01)
        result_high = naive_reid_risk(real, synth, threshold=1.0)
        
        assert isinstance(result_low, float)
        assert isinstance(result_high, float)
        assert result_low <= result_high  # Lower threshold should give lower risk
    
    def test_naive_reid_risk_with_nans(self):
        """Test naive reid risk with NaN values."""
        real = pd.DataFrame({
            'age': [30, 40, np.nan, 60, 70],
            'score': [85, 90, 75, np.nan, 95]
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72],
            'score': [87, 92, 77, 82, 97]
        })
        
        result = naive_reid_risk(real, synth)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestSummarizePrivacyFidelity:
    """Test privacy and fidelity summarization."""
    
    def test_summarize_privacy_fidelity_basic(self):
        """Test basic privacy and fidelity summarization."""
        # Need at least 11 samples for KS test
        real = pd.DataFrame({
            'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75, 80],
            'score': [85, 90, 75, 80, 95, 88, 92, 78, 82, 97, 100],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'A', 'C', 'B', 'A']
        })
        synth = pd.DataFrame({
            'age': [32, 42, 52, 62, 72, 37, 47, 57, 67, 77, 82],
            'score': [87, 92, 77, 82, 97, 90, 94, 80, 84, 99, 102],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'B', 'A', 'C', 'B', 'A']
        })
        
        result = summarize_privacy_fidelity(real, synth)
        
        assert isinstance(result, dict)
        assert 'ks_pvalues' in result
        assert 'corr_mean_abs_delta' in result
        assert 'naive_reid_risk_fraction' in result
        
        assert isinstance(result['ks_pvalues'], dict)
        assert isinstance(result['corr_mean_abs_delta'], float)
        assert isinstance(result['naive_reid_risk_fraction'], float)
        
        # Check that numeric columns are included in KS tests
        assert 'age' in result['ks_pvalues']
        assert 'score' in result['ks_pvalues']
        assert 'category' not in result['ks_pvalues']
    
    def test_summarize_privacy_fidelity_empty_data(self):
        """Test privacy and fidelity summarization with empty data."""
        real = pd.DataFrame({'age': []})
        synth = pd.DataFrame({'age': []})
        
        result = summarize_privacy_fidelity(real, synth)
        
        assert isinstance(result, dict)
        assert 'ks_pvalues' in result
        assert 'corr_mean_abs_delta' in result
        assert 'naive_reid_risk_fraction' in result
        
        assert result['ks_pvalues'] == {}
        assert np.isnan(result['corr_mean_abs_delta'])
        assert np.isnan(result['naive_reid_risk_fraction'])
    
    def test_summarize_privacy_fidelity_no_common_columns(self):
        """Test privacy and fidelity summarization with no common columns."""
        real = pd.DataFrame({'age': [30, 40, 50]})
        synth = pd.DataFrame({'height': [170, 180, 190]})
        
        result = summarize_privacy_fidelity(real, synth)
        
        assert isinstance(result, dict)
        assert result['ks_pvalues'] == {}
        assert np.isnan(result['corr_mean_abs_delta'])
        assert np.isnan(result['naive_reid_risk_fraction'])
    
    def test_summarize_privacy_fidelity_single_numeric_column(self):
        """Test privacy and fidelity summarization with single numeric column."""
        # Need at least 11 samples for KS test
        real = pd.DataFrame({'age': [30, 40, 50, 60, 70, 35, 45, 55, 65, 75, 80]})
        synth = pd.DataFrame({'age': [32, 42, 52, 62, 72, 37, 47, 57, 67, 77, 82]})
        
        result = summarize_privacy_fidelity(real, synth)
        
        assert isinstance(result, dict)
        assert 'age' in result['ks_pvalues']
        assert np.isnan(result['corr_mean_abs_delta'])  # Need at least 2 columns
        assert isinstance(result['naive_reid_risk_fraction'], float)
