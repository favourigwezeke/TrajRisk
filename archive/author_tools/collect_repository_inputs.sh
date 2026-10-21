#!/usr/bin/env bash
set -euo pipefail
python - "$@" <<'PYEOF'
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile

print('\n===== TRAJRISK REPOSITORY CHECK 1: SOURCES / ENVIRONMENT / INPUT INVENTORY =====')

def is_project(p):
    return p.is_dir() and (p/'scripts').is_dir() and ((p/'Data').exists() or (p/'data').exists() or (p/'results').exists())

explicit = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('TRAJRISK_ROOT')
if explicit:
    project = Path(explicit).expanduser().resolve()
    if not is_project(project):
        raise SystemExit('The supplied folder must contain scripts/ and Data/, data/, or results/.')
else:
    candidates = [Path.cwd(), *Path.cwd().parents, Path.home()/'TrajRisk']
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if is_project(p)))
    if len(candidates) != 1:
        raise SystemExit('Run this from your TrajRisk folder, or pass that folder as the first argument.')
    project = candidates[0]

stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
out = project/'results'/('repository_check1_'+stamp)
out.mkdir(parents=True, exist_ok=False)
payload = out/'payload'
payload.mkdir()

expected = {
 'scripts/week5_transition_matrices.py':'2be777dd0c2704f3f476b8f50fe3a63f79ff8f6a61eb07119d2edc3f2aa56329',
 'scripts/week6_gnn_architecture.py':'c4e8dba6221ff80b3250d87baeb82769b338f6d5e92c2f04afaf3c2565ed319d',
 'scripts/week7_full_training.py':'4ceba15a846619db37019532ad8c779bfb6339fa9488fdf03829bd541d7177f7',
 'scripts/week8_baselines_ablations.py':'bf0c4cefed0471fcc444fa1e75f0d7d124c8bbaf760bc6f3df1d2052c2e100f1',
 'scripts/week9a_mouse_cross_dataset_final.py':'7bbae518f6e464abd552b88473a44ca9ca17df78dec864e18911e66f68fe8922',
 'scripts/week9b_human_cross_species.py':'34db603202f0892602a2faf026ca7d76cbd7d3a76a8b1983a2864cb9f2a014d3'
}
source_extensions = {'.py','.r','.rmd','.qmd','.ipynb','.sh','.jl','.yaml','.yml','.json','.toml','.ini','.cfg'}
excluded_dirs = {'.git','.venv','venv','env','envs','__pycache__','node_modules','.ipynb_checkpoints',
                 'site-packages','renv','packrat','raw','processed','models','figures','supplementary'}
secret_patterns = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{16,}'),
    re.compile(r'(?im)^\s*(?:export\s+)?(?:[A-Za-z0-9_]*(?:api_key|access_token|client_secret|password)[A-Za-z0-9_]*)\s*=\s*[\"\x27]([^\"\x27\n]{10,})[\"\x27]')
]
records, omitted, errors = [], [], []

def digest(data):
    return hashlib.sha256(data).hexdigest()

roots = [project/x for x in ['scripts','notebooks','src','config','configs','metadata'] if (project/x).is_dir()]
for source_root in roots:
    for here, dirs, files in os.walk(source_root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs and not d.startswith('.') and not (Path(here)/d).is_symlink())
        for name in sorted(files):
            p = Path(here)/name
            if p.is_symlink() or p.suffix.lower() not in source_extensions or name.startswith('.'):
                continue
            rel = p.relative_to(project).as_posix()
            try:
                before = p.stat()
                if before.st_size > 32*1024*1024:
                    omitted.append({'path':rel,'reason':'Source larger than 32 MiB; not copied'})
                    continue
                raw = p.read_bytes()
                text = raw.decode('utf-8')
                if any(pattern.search(text) for pattern in secret_patterns):
                    omitted.append({'path':rel,'reason':'Potential embedded credential; file omitted, value never printed'})
                    continue
                exported = raw
                notebook_cells = None
                if p.suffix.lower() == '.ipynb':
                    nb = json.loads(text)
                    cells = []
                    for c in nb.get('cells',[]):
                        cell = {'cell_type':c['cell_type'],'metadata':{},'source':c.get('source',[])}
                        if cell['cell_type'] == 'code':
                            cell.update(execution_count=None,outputs=[])
                        cells.append(cell)
                    notebook_cells = sum(c['cell_type']=='code' for c in cells)
                    nb = {'cells':cells,'metadata':{'kernelspec':nb.get('metadata',{}).get('kernelspec',{})},
                          'nbformat':nb.get('nbformat',4),'nbformat_minor':nb.get('nbformat_minor',5)}
                    exported = (json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode()
                after = p.stat()
                if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                    raise RuntimeError('Input changed during collection; retry after research writes stop')
                dest = payload/'sources'/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(exported)
                record = {'path':rel,'bytes':len(raw),'original_sha256':digest(raw),'export_sha256':digest(exported),
                          'legacy_named':bool(re.search(r'(?i)(?:^|[_ .-])(old|backup|bak)(?:[_ .-]|$)',p.stem)),
                          'notebook_code_cells':notebook_cells,'input_unchanged':True}
                if rel in expected:
                    record['matches_previously_audited_source'] = digest(raw)==expected[rel]
                records.append(record)
            except Exception as exc:
                errors.append({'path':rel,'error_type':type(exc).__name__,'reason':str(exc)[:180]})

# Inventory model/data/result assets without loading, unpickling, hashing, or running them.
inventory = []
asset_suffixes = {'.h5ad','.rds','.pt','.pth','.pkl','.joblib','.npz','.npy','.csv','.json','.log','.txt','.pdf'}
for top in ['Data','data','models','results']:
    root = project/top
    if not root.is_dir():
        continue
    for here, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded_dirs-{'models','figures','raw','processed'}
                         and not d.startswith('.') and not (Path(here)/d).is_symlink()
                         and not re.match(r'(?i)(repository_check|equation_check|.*_evidence_20|.*_audit_20)',d))
        for name in sorted(files):
            p = Path(here)/name
            if p.is_symlink() or p.suffix.lower() not in asset_suffixes:
                continue
            try:
                st=p.stat()
                inventory.append({'path':p.relative_to(project).as_posix(),'bytes':st.st_size})
            except OSError:
                errors.append({'path':p.relative_to(project).as_posix(),'error_type':'OSError','reason':'Cannot stat asset'})

packages = {}
for dist in metadata.distributions():
    name=dist.metadata.get('Name')
    if name:
        packages[name]=dist.version
environment = {'python':sys.version.split()[0],'packages':dict(sorted(packages.items(),key=lambda x:x[0].lower())),
               'r_session':'Already user-confirmed; not rerun by this collector'}
conda = shutil.which('conda')
if conda:
    try:
        run=subprocess.run([conda,'list','--json'],capture_output=True,text=True,timeout=45)
        if run.returncode==0:
            environment['conda_packages']=[{k:d[k] for k in ['name','version','build_string'] if k in d} for d in json.loads(run.stdout)]
        else:
            environment['conda_export_note']='Conda list failed; installed Python package versions were still recorded.'
    except Exception as exc:
        environment['conda_export_note']=type(exc).__name__+'; installed Python package versions were still recorded.'

for name, obj in [('SOURCE_MANIFEST.json',records),('ASSET_INVENTORY.json',inventory),
                  ('ENVIRONMENT.json',environment),('OMITTED_FILES.json',omitted),('READ_ERRORS.json',errors)]:
    (payload/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
core = {rel:next((r.get('matches_previously_audited_source') for r in records if r['path']==rel),None) for rel in expected}
summary={'source_files_collected':len(records),'notebook_code_cells':sum(r['notebook_code_cells'] or 0 for r in records),
         'inventory_assets':len(inventory),'omitted_files':len(omitted),'read_errors':len(errors),
         'core_source_matches':core,'research_scripts_executed':False,'existing_research_files_modified':False}
(payload/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(payload/'README.txt').write_text('Private source-collection packet for repository preparation. Research code was not executed.\n'
    'Notebooks contain source cells with outputs and execution counts removed. Original hashes are retained.\n'
    'Large expression matrices and checkpoints are listed only; their contents were not copied or unpickled.\n'
    'Potential embedded credentials are omitted by a heuristic scan. Review before any public release.\n')
archive = out/'TrajRisk_Repository_Check_1.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(payload.rglob('*')):
        if p.is_file():
            z.write(p,p.relative_to(payload))
print('Source files collected:',len(records))
print('Notebook code cells:',summary['notebook_code_cells'])
print('Previously audited core source matches:',sum(v is True for v in core.values()),'/',len(core))
print('Model/data/result assets inventoried:',len(inventory))
print('Omitted files:',len(omitted),'| Read errors:',len(errors))
print('No training, inference, figure regeneration, or R-session rerun was performed.')
print('\nUPLOAD THIS ZIP:',archive)
print('ZIP size: %.2f MiB' % (archive.stat().st_size/1024**2))
PYEOF
