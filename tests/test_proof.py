# tests/test_proof.py
import json
from verisynth.lineage import merkle_root  # example helper

def test_merkle_root_consistency(tmp_path):
    # Create some test hashes (simulating row hashes)
    data = ["row1", "row2", "row3"]
    # Convert to hashes as merkle_root expects a list of hash strings
    import hashlib
    hashes = [hashlib.sha256(row.encode()).hexdigest() for row in data]
    
    root1 = merkle_root(hashes)
    root2 = merkle_root(hashes)
    assert root1 == root2
