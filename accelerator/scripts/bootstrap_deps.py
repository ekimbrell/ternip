"""Fetch only pinned, hash-verified BaseJump sources used by the regression."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
p=argparse.ArgumentParser();p.add_argument('--output',default='build/basejump_stl');a=p.parse_args()
lock=json.loads((Path(__file__).resolve().parents[1]/'locks/basejump.json').read_text())
root=Path(a.output)
for name in lock['files']:
    path=root/name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==lock['sha256'][name]: continue
    url=f"https://raw.githubusercontent.com/{lock['repository']}/{lock['revision']}/{name}"
    with urlopen(url,timeout=60) as response: data=response.read()
    # Sources were archived with LF line endings.
    data=data.replace(b'\r\n',b'\n').rstrip()+b'\n'
    if hashlib.sha256(data).hexdigest()!=lock['sha256'][name]: raise ValueError(f'Dependency hash mismatch: {name}')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
print('Pinned BaseJump dependency ready')
