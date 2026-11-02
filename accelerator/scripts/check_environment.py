"""Audit the pinned Alveo build stack without installing or changing tools."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ternip_accel.environment import audit

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--tool-root', type=Path)
p.add_argument('--platform-file', type=Path)
a = p.parse_args()
result = audit(json.loads(a.target.read_text()), a.output, a.tool_root, a.platform_file)
print(result['status'])
for reason in result['blockers']:
    print('- ' + reason)
raise SystemExit(0 if not result['blockers'] else 1)
