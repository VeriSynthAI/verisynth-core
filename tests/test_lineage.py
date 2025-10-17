# tests/test_lineage.py
import pytest
import pandas as pd
import hashlib
import json
import tempfile
import os
from verisynth.lineage import (
    sha256_bytes,
    hash_row,
    merkle_root,
    dataframe_merkle,
    file_hash
)


class TestSHA256Bytes:
    """Test SHA256 bytes functionality."""
    
    def test_sha256_bytes_basic(self):
        """Test basic SHA256 bytes functionality."""
        data = b"hello world"
        result = sha256_bytes(data)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
        assert all(c in '0123456789abcdef' for c in result)
    
    def test_sha256_bytes_empty(self):
        """Test SHA256 bytes with empty input."""
        result = sha256_bytes(b"")
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_sha256_bytes_deterministic(self):
        """Test that SHA256 bytes is deterministic."""
        data = b"test data"
        result1 = sha256_bytes(data)
        result2 = sha256_bytes(data)
        assert result1 == result2
    
    def test_sha256_bytes_different_inputs(self):
        """Test that different inputs produce different hashes."""
        data1 = b"hello"
        data2 = b"world"
        result1 = sha256_bytes(data1)
        result2 = sha256_bytes(data2)
        assert result1 != result2


class TestHashRow:
    """Test row hashing functionality."""
    
    def test_hash_row_basic(self):
        """Test basic row hashing."""
        row = pd.Series({'age': 30, 'name': 'Alice', 'score': 85.5})
        result = hash_row(row)
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
    
    def test_hash_row_deterministic(self):
        """Test that row hashing is deterministic."""
        row = pd.Series({'age': 30, 'name': 'Alice'})
        result1 = hash_row(row)
        result2 = hash_row(row)
        assert result1 == result2
    
    def test_hash_row_order_independent(self):
        """Test that row hashing is order independent."""
        row1 = pd.Series({'age': 30, 'name': 'Alice'})
        row2 = pd.Series({'name': 'Alice', 'age': 30})
        result1 = hash_row(row1)
        result2 = hash_row(row2)
        assert result1 == result2  # Should be same due to sort_keys=True
    
    def test_hash_row_with_nans(self):
        """Test row hashing with NaN values."""
        row = pd.Series({'age': 30, 'score': float('nan'), 'name': 'Alice'})
        result = hash_row(row)
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_hash_row_different_values(self):
        """Test that different row values produce different hashes."""
        row1 = pd.Series({'age': 30, 'name': 'Alice'})
        row2 = pd.Series({'age': 31, 'name': 'Alice'})
        result1 = hash_row(row1)
        result2 = hash_row(row2)
        assert result1 != result2


class TestMerkleRoot:
    """Test Merkle root functionality."""
    
    def test_merkle_root_empty(self):
        """Test Merkle root with empty list."""
        result = merkle_root([])
        assert isinstance(result, str)
        assert len(result) == 64
        # Should be hash of empty bytes
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
    
    def test_merkle_root_single_hash(self):
        """Test Merkle root with single hash."""
        hashes = ["a" * 64]  # Single 64-character hex string
        result = merkle_root(hashes)
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_merkle_root_multiple_hashes(self):
        """Test Merkle root with multiple hashes."""
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        result = merkle_root(hashes)
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_merkle_root_odd_number(self):
        """Test Merkle root with odd number of hashes."""
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        result = merkle_root(hashes)
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_merkle_root_deterministic(self):
        """Test that Merkle root is deterministic."""
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        result1 = merkle_root(hashes)
        result2 = merkle_root(hashes)
        assert result1 == result2
    
    def test_merkle_root_different_order(self):
        """Test that Merkle root is order dependent."""
        hashes1 = ["a" * 64, "b" * 64, "c" * 64]
        hashes2 = ["c" * 64, "b" * 64, "a" * 64]
        result1 = merkle_root(hashes1)
        result2 = merkle_root(hashes2)
        assert result1 != result2  # Order should matter


class TestDataframeMerkle:
    """Test DataFrame Merkle functionality."""
    
    def test_dataframe_merkle_basic(self):
        """Test basic DataFrame Merkle functionality."""
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result = dataframe_merkle(df)
        
        assert isinstance(result, dict)
        assert 'row_hashes_count' in result
        assert 'merkle_root' in result
        
        assert result['row_hashes_count'] == 3
        assert isinstance(result['merkle_root'], str)
        assert len(result['merkle_root']) == 64
    
    def test_dataframe_merkle_empty(self):
        """Test DataFrame Merkle with empty DataFrame."""
        df = pd.DataFrame()
        result = dataframe_merkle(df)
        
        assert isinstance(result, dict)
        assert result['row_hashes_count'] == 0
        assert isinstance(result['merkle_root'], str)
        assert len(result['merkle_root']) == 64
    
    def test_dataframe_merkle_single_row(self):
        """Test DataFrame Merkle with single row."""
        df = pd.DataFrame({'age': [30], 'name': ['Alice']})
        result = dataframe_merkle(df)
        
        assert result['row_hashes_count'] == 1
        assert isinstance(result['merkle_root'], str)
        assert len(result['merkle_root']) == 64
    
    def test_dataframe_merkle_deterministic(self):
        """Test that DataFrame Merkle is deterministic."""
        df = pd.DataFrame({
            'age': [30, 40, 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result1 = dataframe_merkle(df)
        result2 = dataframe_merkle(df)
        assert result1 == result2
    
    def test_dataframe_merkle_with_nans(self):
        """Test DataFrame Merkle with NaN values."""
        df = pd.DataFrame({
            'age': [30, float('nan'), 50],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        result = dataframe_merkle(df)
        
        assert result['row_hashes_count'] == 3
        assert isinstance(result['merkle_root'], str)
        assert len(result['merkle_root']) == 64


class TestFileHash:
    """Test file hashing functionality."""
    
    def test_file_hash_basic(self, tmp_path):
        """Test basic file hashing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        result = file_hash(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex string length
    
    def test_file_hash_empty_file(self, tmp_path):
        """Test file hashing with empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = file_hash(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64
        # Should be hash of empty bytes
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected
    
    def test_file_hash_deterministic(self, tmp_path):
        """Test that file hashing is deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        result1 = file_hash(str(test_file))
        result2 = file_hash(str(test_file))
        assert result1 == result2
    
    def test_file_hash_different_content(self, tmp_path):
        """Test that different file content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        
        result1 = file_hash(str(file1))
        result2 = file_hash(str(file2))
        assert result1 != result2
    
    def test_file_hash_large_file(self, tmp_path):
        """Test file hashing with larger file."""
        test_file = tmp_path / "large.txt"
        content = "x" * 10000  # 10KB of data
        test_file.write_text(content)
        
        result = file_hash(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64
    
    def test_file_hash_nonexistent_file(self):
        """Test file hashing with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_hash("nonexistent_file.txt")
    
    def test_file_hash_binary_content(self, tmp_path):
        """Test file hashing with binary content."""
        test_file = tmp_path / "binary.bin"
        binary_content = b'\x00\x01\x02\x03\xff\xfe\xfd\xfc'
        test_file.write_bytes(binary_content)
        
        result = file_hash(str(test_file))
        
        assert isinstance(result, str)
        assert len(result) == 64
