# tests/test_verify.py
import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from verisynth.verify import (
    file_sha256,
    row_hash,
    merkle_root,
    verify_proof
)


class TestFileSHA256:
    """Test file SHA256 functionality."""
    
    def test_file_sha256_basic(self, tmp_path):
        """Test basic file SHA256 functionality."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        result = file_sha256(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
    
    def test_file_sha256_empty_file(self, tmp_path):
        """Test file SHA256 with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = file_sha256(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64
        # Should be hash of empty bytes
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
    
    def test_file_sha256_deterministic(self, tmp_path):
        """Test that file SHA256 is deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result1 = file_sha256(str(test_file))
        result2 = file_sha256(str(test_file))
        assert result1 == result2
    
    def test_file_sha256_different_content(self, tmp_path):
        """Test that different file content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        
        result1 = file_sha256(str(file1))
        result2 = file_sha256(str(file2))
        assert result1 != result2
    
    def test_file_sha256_large_file(self, tmp_path):
        """Test file SHA256 with larger file."""
        test_file = tmp_path / "large.txt"
        content = "x" * 1000000  # 1MB of data
        test_file.write_text(content)
        
        result = file_sha256(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_file_sha256_nonexistent_file(self):
        """Test file SHA256 with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_sha256("nonexistent_file.txt")


class TestRowHash:
    """Test row hashing functionality."""
    
    def test_row_hash_basic(self):
        """Test basic row hashing."""
        row = pd.Series({'age': 30, 'name': 'Alice', 'score': 85.5})
        result = row_hash(row)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
    
    def test_row_hash_deterministic(self):
        """Test that row hashing is deterministic."""
        row = pd.Series({'age': 30, 'name': 'Alice'})
        result1 = row_hash(row)
        result2 = row_hash(row)
        assert result1 == result2
    
    def test_row_hash_order_independent(self):
        """Test that row hashing is order independent."""
        row1 = pd.Series({'age': 30, 'name': 'Alice'})
        row2 = pd.Series({'name': 'Alice', 'age': 30})
        result1 = row_hash(row1)
        result2 = row_hash(row2)
        assert result1 == result2  # Should be same due to sort_keys=True
    
    def test_row_hash_with_nans(self):
        """Test row hashing with NaN values."""
        row = pd.Series({'age': 30, 'score': float('nan'), 'name': 'Alice'})
        result = row_hash(row)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_row_hash_different_values(self):
        """Test that different row values produce different hashes."""
        row1 = pd.Series({'age': 30, 'name': 'Alice'})
        row2 = pd.Series({'age': 31, 'name': 'Alice'})
        result1 = row_hash(row1)
        result2 = row_hash(row2)
        assert result1 != result2


class TestMerkleRoot:
    """Test Merkle root functionality."""
    
    def test_merkle_root_empty(self):
        """Test Merkle root with empty DataFrame."""
        df = pd.DataFrame()
        result = merkle_root(df)
        
        assert isinstance(result, str)
        assert len(result) == 64
        # Should be hash of empty bytes
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
    
    def test_merkle_root_single_row(self):
        """Test Merkle root with single row."""
        df = pd.DataFrame({'age': [30], 'name': ['Alice']})
        result = merkle_root(df)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_merkle_root_multiple_rows(self):
        """Test Merkle root with multiple rows."""
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result = merkle_root(df)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_merkle_root_deterministic(self):
        """Test that Merkle root is deterministic."""
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result1 = merkle_root(df)
        result2 = merkle_root(df)
        assert result1 == result2
    
    def test_merkle_root_with_nans(self):
        """Test Merkle root with NaN values."""
        df = pd.DataFrame({
            'age': [30, float('nan'), 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result = merkle_root(df)
        
        assert isinstance(result, str)
        assert len(result) == 64


class TestVerifyIntegration:
    """Test verification integration functionality."""
    
    def test_proof_verification_integration(self, tmp_path):
        """Test end-to-end proof verification."""
        # Create test data
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_df = pd.DataFrame({
            'age': [30, 40, 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        output_df = pd.DataFrame({
            'age': [32, 42, 52],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        input_df.to_csv(input_file, index=False)
        output_df.to_csv(output_file, index=False)
        
        # Create proof
        proof = {
            "input": {
                "path": str(input_file),
                "file_sha256": file_sha256(str(input_file)),
                "merkle_root": merkle_root(input_df)
            },
            "output": {
                "path": str(output_file),
                "file_sha256": file_sha256(str(output_file)),
                "merkle_root": merkle_root(output_df)
            }
        }
        
        # Verify proof
        assert file_sha256(proof["input"]["path"]) == proof["input"]["file_sha256"]
        assert file_sha256(proof["output"]["path"]) == proof["output"]["file_sha256"]
        
        input_verification_df = pd.read_csv(proof["input"]["path"])
        output_verification_df = pd.read_csv(proof["output"]["path"])
        
        assert merkle_root(input_verification_df) == proof["input"]["merkle_root"]
        assert merkle_root(output_verification_df) == proof["output"]["merkle_root"]
    
    def test_proof_verification_failure(self, tmp_path):
        """Test proof verification failure scenarios."""
        # Create test data
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_df = pd.DataFrame({'age': [30, 40, 50]})
        output_df = pd.DataFrame({'age': [32, 42, 52]})
        
        input_df.to_csv(input_file, index=False)
        output_df.to_csv(output_file, index=False)
        
        # Create proof with wrong hash
        proof = {
            "input": {
                "path": str(input_file),
                "file_sha256": "wrong_hash",
                "merkle_root": merkle_root(input_df)
            },
            "output": {
                "path": str(output_file),
                "file_sha256": file_sha256(str(output_file)),
                "merkle_root": merkle_root(output_df)
            }
        }
        
        # Verify that verification fails
        assert file_sha256(proof["input"]["path"]) != proof["input"]["file_sha256"]
        assert file_sha256(proof["output"]["path"]) == proof["output"]["file_sha256"]


class TestVerifyProof:
    """Test verify_proof function."""
    
    def test_verify_proof_success(self, tmp_path):
        """Test successful proof verification."""
        # Create test data
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_df = pd.DataFrame({'age': [30, 40, 50]})
        output_df = pd.DataFrame({'age': [32, 42, 52]})
        
        input_df.to_csv(input_file, index=False)
        output_df.to_csv(output_file, index=False)
        
        # Create proof
        proof = {
            "input": {
                "path": str(input_file),
                "file_sha256": file_sha256(str(input_file)),
                "merkle_root": merkle_root(input_df)
            },
            "output": {
                "path": str(output_file),
                "file_sha256": file_sha256(str(output_file)),
                "merkle_root": merkle_root(output_df)
            }
        }
        
        proof_file = tmp_path / "proof.json"
        with open(proof_file, 'w') as f:
            json.dump(proof, f)
        
        # Verify proof
        verify_proof(str(proof_file))
    
    def test_verify_proof_failure(self, tmp_path):
        """Test proof verification failure."""
        # Create test data
        input_file = tmp_path / "input.csv"
        output_file = tmp_path / "output.csv"
        
        input_df = pd.DataFrame({'age': [30, 40, 50]})
        output_df = pd.DataFrame({'age': [32, 42, 52]})
        
        input_df.to_csv(input_file, index=False)
        output_df.to_csv(output_file, index=False)
        
        # Create proof with wrong hash
        proof = {
            "input": {
                "path": str(input_file),
                "file_sha256": "wrong_hash",
                "merkle_root": merkle_root(input_df)
            },
            "output": {
                "path": str(output_file),
                "file_sha256": file_sha256(str(output_file)),
                "merkle_root": merkle_root(output_df)
            }
        }
        
        proof_file = tmp_path / "proof.json"
        with open(proof_file, 'w') as f:
            json.dump(proof, f)
        
        # Verify that verification fails
        with pytest.raises(AssertionError):
            verify_proof(str(proof_file))
    
    def test_verify_proof_nonexistent_file(self):
        """Test proof verification with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            verify_proof("nonexistent_proof.json")
