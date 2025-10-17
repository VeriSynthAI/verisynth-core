\
import hashlib
import json
import pandas as pd
from typing import Dict, Any, List

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hash_row(row: pd.Series) -> str:
    # stable JSON for row values
    obj = row.to_dict()
    j = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(j)

def merkle_root(hashes: List[str]) -> str:
    if not hashes:
        return sha256_bytes(b"")
    level = [bytes.fromhex(h) for h in hashes]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if i+1 < len(level) else left
            nxt.append(hashlib.sha256(left + right).digest())
        level = nxt
    return level[0].hex()

def dataframe_merkle(df: pd.DataFrame) -> Dict[str, Any]:
    row_hashes = [hash_row(r) for _, r in df.iterrows()]
    root = merkle_root(row_hashes)
    return {"row_hashes_count": len(row_hashes), "merkle_root": root}

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
