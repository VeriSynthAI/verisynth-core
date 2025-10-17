import json, hashlib, pandas as pd

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def row_hash(row):
    return hashlib.sha256(
        json.dumps(row.to_dict(), sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()

def merkle_root(df):
    hashes = [bytes.fromhex(row_hash(r)) for _, r in df.iterrows()]
    if not hashes: return hashlib.sha256(b'').hexdigest()
    level = hashes
    while len(level) > 1:
        nxt=[]
        for i in range(0,len(level),2):
            a=level[i]; b=level[i+1] if i+1<len(level) else a
            nxt.append(hashlib.sha256(a+b).digest())
        level = nxt
    return level[0].hex()

proof = json.load(open("out/proof.json"))
assert file_sha256(proof["input"]["path"])  == proof["input"]["file_sha256"]
assert file_sha256(proof["output"]["path"]) == proof["output"]["file_sha256"]

rin = pd.read_csv(proof["input"]["path"])
rout = pd.read_csv(proof["output"]["path"])
assert merkle_root(rin)  == proof["input"]["merkle_root"]
assert merkle_root(rout) == proof["output"]["merkle_root"]
print("OK: hashes and Merkle roots verified.")
