"""Immutable, verified artifact acquisition; never execute remote model code."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, Request

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(4*1024**2),b""): h.update(chunk)
    return h.hexdigest()

def verify(root, lock):
    root = Path(root)
    for name,info in lock["files"].items():
        p = root/name
        if p.stat().st_size != info["size"] or sha256(p) != info["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")

def fetch(root, lock):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    for name,info in lock["files"].items():
        p = root/name
        if p.exists() and p.stat().st_size==info["size"] and sha256(p)==info["sha256"]: continue
        url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{name}"
        tmp = root/(name+".partial")
        print(f"Downloading {name} ({info['size']} bytes)",flush=True)
        for attempt in range(3):
            offset = tmp.stat().st_size if tmp.exists() else 0
            request = Request(url, headers={"Range":f"bytes={offset}-"} if offset else {})
            with urlopen(request,timeout=120) as response:
                append = response.status == 206 and offset > 0
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resume range")
                with tmp.open("ab" if append else "wb") as out:
                    while True:
                        chunk = response.read(4*1024**2)
                        if not chunk: break
                        out.write(chunk)
            if tmp.stat().st_size == info["size"]: break
            print(f"Incomplete download: {tmp.stat().st_size}/{info['size']}; resuming",flush=True)
        if tmp.stat().st_size != info["size"] or sha256(tmp) != info["sha256"]:
            raise ValueError(f"Downloaded artifact hash mismatch: {name}")
        tmp.replace(p)
    verify(root,lock)

def load_checkpoint(root, lock):
    verify(root,lock)
    from safetensors import safe_open
    # torch conversion handles BF16, which NumPy does not natively expose.
    tensors = {}
    with safe_open(str(Path(root)/"model.safetensors"),framework="pt",device="cpu") as f:
        for name in f.keys(): tensors[name] = f.get_tensor(name).float().numpy()
    return json.loads((Path(root)/"config.json").read_text()),tensors
