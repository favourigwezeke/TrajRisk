"""Validate the supplied publication assets without running the analyses."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'metadata/file_manifest.json').read_text())
failures = []
for item in manifest['files']:
    path=root/item['path']
    if not path.is_file():
        failures.append('Missing: '+item['path'])
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        failures.append('Changed: '+item['path'])

paper = root/'manuscript'
reference_keys = re.findall(r'\\bibitem\{([^}]+)\}',(paper/'references.tex').read_text())
active_tex = list(paper.glob('*.tex')) + [p for d in ['sections','legends','tables','data','figures'] for p in (paper/d).rglob('*.tex')]
cited=set()
for p in active_tex:
    for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',p.read_text()):
        cited.update(k.strip() for k in group.split(','))
if len(reference_keys)!=126 or len(reference_keys)!=len(set(reference_keys)) or set(reference_keys)!=cited:
    failures.append('Reference matching differs from the 126-entry canonical bibliography')
methods=(paper/'sections/methods.tex').read_text()
equations=len(re.findall(r'\\begin\{equation\}',methods))
if equations!=26:
    failures.append('Numbered equation count differs from 26')
data_files=json.loads((root/'metadata/supplementary_data_checks.json').read_text())['files']
for item in data_files:
    path=paper/'data'/item['filename']
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        rows=list(reader)
        if len(rows)!=item['rows'] or reader.fieldnames!=item['columns']:
            failures.append('CSV structure differs: '+item['filename'])
for p in (root/'scripts').glob('*.py'):
    try:
        ast.parse(p.read_text())
    except SyntaxError:
        failures.append('Python syntax: '+p.name)
if failures:
    print('\n'.join(failures),file=sys.stderr)
    raise SystemExit(1)
print('Supplied file hashes: %d match' % len(manifest['files']))
print('References: 126 entries, all cited exactly once in the bibliography')
print('Numbered equations: 26')
print('Data S1–S5: expected columns and record counts match')
print('This checks the supplied assets; it does not rerun research analyses.')
