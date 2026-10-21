#!/usr/bin/env bash
set -euo pipefail
python - "$@" <<'PYEOF'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
import zipfile

import numpy as np
import pandas as pd
import h5py

print('\n===== TRAJRISK REPOSITORY CHECK 2: FROZEN INPUTS / COMPACT ARRAYS =====',flush=True)
project=Path(sys.argv[1] if len(sys.argv)>1 else os.environ.get('TRAJRISK_ROOT',os.getcwd())).expanduser().resolve()
if not (project/'scripts').is_dir() or not (project/'results').is_dir():
    raise SystemExit('Run this from your TrajRisk directory, or pass that directory as the first argument.')
upper,lower=project/'Data',project/'data'
case_info={'Data_exists':upper.is_dir(),'data_exists':lower.is_dir(),'same_directory':None}
if upper.is_dir() and lower.is_dir():
    case_info['same_directory']=os.path.samefile(upper,lower)
data_root=upper if upper.is_dir() else lower
if not data_root.is_dir():
    raise SystemExit('Neither Data/ nor data/ exists in the supplied project.')
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
out=project/'results'/('repository_check2_'+stamp)
out.mkdir(parents=True,exist_ok=False)
payload=out/'payload';payload.mkdir()
copied=[];compact=[];missing=[];issues=[];states=[]
copied_targets=set()
MAX_FILE=96*1024**2
MAX_TOTAL=512*1024**2
total=0

def json_write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def stat_key(p):
    st=p.stat();return (st.st_size,st.st_mtime_ns)

def add_file(p,rel,required=False):
    global total
    if rel in copied_targets:
        issues.append({'path':rel,'reason':'Duplicate destination from a second source directory; not overwritten'});return
    if not p.is_file():
        missing.append({'path':rel,'required':required});return
    if p.is_symlink():
        issues.append({'path':rel,'reason':'Symlinked input not copied; review target explicitly'});return
    before=stat_key(p)
    if before[0]>MAX_FILE or total+before[0]>MAX_TOTAL:
        issues.append({'path':rel,'reason':'Collection size cap exceeded; not copied'});return
    dest=payload/'originals'/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256()
    with p.open('rb') as src,dest.open('wb') as dst:
        while True:
            block=src.read(1024*1024)
            if not block:break
            h.update(block);dst.write(block)
    after=stat_key(p)
    if before!=after:
        dest.unlink()
        issues.append({'path':rel,'reason':'Input changed during collection; exported copy removed'});return
    states.append((p,before))
    total+=before[0]
    copied.append({'path':rel,'bytes':before[0],'sha256':h.hexdigest(),'input_unchanged':True})
    copied_targets.add(rel)
    if before[0]>20*1024**2:
        print('  Copied %s (%.1f MiB)'%(rel,before[0]/1024**2),flush=True)

required_processed=[
 'T_Scale1_AcuteStress.npz','T_Scale2_PeakInjury.npz','T_Scale3_FateDecision.npz','T_Scale4_Resolution.npz',
 'multiscale_graphs.pt','week6_node_features.pt','week6_training_data.pt','week6_xgboost_risk.npy',
 'week7_predictions.csv','week7_scale_embeddings.npy','walesky_cell_index.csv','pseudotime_walesky.csv',
 'mouse_human_orthologs_biomart.csv']
for name in required_processed:
    add_file(data_root/'processed'/name,'Data/processed/'+name,True)
required_models=['week7_trajrisk_FINAL.pt','week9b_trajrisk_cross_species.pt',
                 'week6_xgboost_model.pkl','week9b_cross_species_seeder.pkl']
for name in required_models:
    add_file(project/'models'/name,'models/'+name,True)
for name in ['week7_trajrisk_BEST_AUROC.pt','week7_trajrisk_BEST.pt']:
    add_file(project/'models'/name,'models/'+name,False)

# Numerical outputs are copied as bytes; figures and experimental histories are not rerun.
prefixes=('bottleneck_','week5_','week6_','week7_','week8_','week9a_','week9b_','week10_','week12_','manuscript_')
for p in sorted((project/'results').iterdir()):
    if p.is_file() and p.suffix.lower() in {'.csv','.json'} and p.name.startswith(prefixes):
        add_file(p,'results/'+p.name)
for folder in ['Supplementary','supplementary']:
    supp=project/folder
    if not supp.is_dir():continue
    if folder=='supplementary' and (project/'Supplementary').is_dir() and os.path.samefile(supp,project/'Supplementary'):continue
    for p in sorted(supp.rglob('*.csv')):
        if re.match(r'^(?:Table_S[1-6][A-G]?(?:_|\.)|Data_S[1-5](?:_|\.))',p.name):
            add_file(p,'Supplementary/'+p.relative_to(supp).as_posix())

def decode_strings(values):
    a=np.asarray(values)
    if a.dtype.kind in {'S','O'}:
        flat=[x.decode('utf-8') if isinstance(x,(bytes,np.bytes_)) else x for x in a.ravel()]
        return np.asarray(flat,dtype=object).reshape(a.shape)
    return a

def column(node):
    if isinstance(node,h5py.Dataset):
        a=decode_strings(node[...])
        if a.ndim!=1:raise ValueError('Expected one-dimensional metadata column')
        return a
    if 'codes' in node and 'categories' in node:
        return pd.Categorical.from_codes(np.asarray(node['codes'][...],dtype=int),categories=column(node['categories']))
    if 'values' in node and 'mask' in node:
        values=pd.Series(column(node['values']))
        values[np.asarray(node['mask'][...],dtype=bool)]=pd.NA
        return values.array
    raise ValueError('Unsupported metadata encoding: '+str(node.attrs.get('encoding-type','unknown')))

def axis_index(group):
    key=group.attrs.get('_index','_index')
    if isinstance(key,bytes):key=key.decode()
    return pd.Index(column(group[key]).astype(str))

def obs_frame(group):
    index=axis_index(group)
    if not index.is_unique:raise ValueError('Observation identifiers are not unique')
    names=group.attrs.get('column-order',[])
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in names]
    df=pd.DataFrame(index=index)
    skipped=[]
    for name in names:
        values=column(group[name])
        # Exclude filesystem-location fields from compact, potentially public metadata.
        if pd.Series(values).astype(str).str.contains(r'/Users/|/workspace/|[A-Za-z]:\\Users\\',regex=True,na=False).any():
            skipped.append(name);continue
        df[name]=values
    df.index.name='cell_id'
    return df,skipped

h5_sources=[
 ('mouse_integration','adata_integrated_with_trajectory.h5ad','X_scvi',True),
 ('cross_species','adata_cross_species_combined.h5ad','X_scvi_cross',True),
 ('benmoshe_qc','benmoshe_hepatocytes.h5ad',None,True),
 ('umbaugh_2h_qc','umbaugh_2h_filtered.h5ad',None,True),
 ('umbaugh_late_qc','umbaugh_48h72h_hepatocytes.h5ad',None,True),
 ('matchett_human_qc','matchett_human_hepatocytes.h5ad',None,True)]
print('  Reading saved metadata and latent coordinates; expression matrices stay on disk.',flush=True)
for key,name,required_latent,required in h5_sources:
    p=data_root/'processed'/name
    if not p.is_file():
        missing.append({'path':'Data/processed/'+name,'required':required});continue
    dest=payload/'compact'/key
    try:
        before=stat_key(p)
        dest.mkdir(parents=True,exist_ok=True)
        with h5py.File(p,'r') as h5:
            obs,skipped=obs_frame(h5['obs'])
            genes=axis_index(h5['var'])
            obs.to_csv(dest/'observations.csv',float_format='%.17g')
            pd.DataFrame({'gene_symbol':genes}).to_csv(dest/'gene_axis.csv',index=False)
            arrays={}
            if required_latent:
                for name_obsm in ['X_scvi','X_scvi_cross','X_umap','X_pca']:
                    if 'obsm' not in h5 or name_obsm not in h5['obsm']:continue
                    slot=h5['obsm'][name_obsm]
                    if not isinstance(slot,h5py.Dataset):continue
                    if len(slot.shape)!=2 or slot.shape[0]!=len(obs) or slot.shape[1]>512:continue
                    if math.prod(slot.shape)*slot.dtype.itemsize>64*1024**2:
                        issues.append({'path':key+'/'+name_obsm,'reason':'Latent/PCA array exceeds compact export cap'});continue
                    a=np.asarray(slot[...])
                    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
                        raise ValueError('Latent coordinate array is non-numeric or non-finite: '+name_obsm)
                    arrays[name_obsm]=a
                if required_latent not in arrays:
                    raise ValueError('Required latent coordinates missing: '+required_latent)
                np.savez_compressed(dest/'coordinates.npz',**arrays)
            record={'key':key,'source':'Data/processed/'+p.name,'source_bytes':before[0],
                    'observations':len(obs),'genes':len(genes),'obs_columns':list(obs.columns),
                    'omitted_filesystem_columns':skipped,
                    'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
                    'expression_matrix_exported':False,'full_source_sha256_computed':False}
        if before!=stat_key(p):raise RuntimeError('Input changed during metadata export')
        states.append((p,before));record['input_unchanged']=True
        json_write(dest/'metadata.json',record);compact.append(record)
        print('  %s: %s observations; %s genes; %s coordinate arrays'%(key,f'{len(obs):,}',f'{len(genes):,}',len(arrays)),flush=True)
    except Exception as exc:
        issues.append({'path':'Data/processed/'+p.name,'reason':type(exc).__name__+': '+str(exc)[:200]})
        if dest.is_dir():shutil.rmtree(dest)

changed=[str(p.relative_to(project)) for p,s in states if stat_key(p)!=s]
if changed:issues.append({'path':'input_preservation','reason':'Inputs changed externally during collection','files':changed})
summary={'date':datetime.now(timezone.utc).isoformat(),'copied_files':len(copied),'copied_bytes':total,
         'compact_objects':len(compact),'case_handling':case_info,
         'missing_required':[x['path'] for x in missing if x['required']],
         'missing_optional':[x['path'] for x in missing if not x['required']],
         'review_items':issues,'research_models_executed':False,'r_session_rerun':False,
         'expression_matrices_copied':False,'inputs_changed_during_collection':changed,
         'large_rf_and_scvi_models_included':False}
json_write(payload/'SUMMARY.json',summary)
json_write(payload/'COPIED_INPUTS.json',copied)
json_write(payload/'COMPACT_OBJECTS.json',compact)
(payload/'README.txt').write_text('Frozen-input collection packet. Original numerical/model files are byte copies.\n'
 'Compact objects contain saved metadata, gene axes and saved latent/PCA/UMAP arrays only.\n'
 'Compact objects cannot substitute for expression matrices in preprocessing, DE or signature calculations.\n'
 'No model was unpickled or executed, no expression matrix was loaded, and no figures or R sessions were rerun.\n'
 'Large RF/scVI models and complete expression matrices remain separate archive inputs.\n')
files=[]
for p in sorted(payload.rglob('*')):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        files.append({'path':p.relative_to(payload).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
json_write(payload/'PACKET_MANIFEST.json',{'files':files})
archive=out/'TrajRisk_Repository_Check_2.zip'
print('  Compressing the collection...',flush=True)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in sorted(payload.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(payload))
print('\nCopied frozen input/result files:',len(copied))
print('Compact metadata/coordinate exports:',len(compact))
print('Data/data refer to the same directory:',case_info['same_directory'])
print('Missing required files:',len(summary['missing_required']))
print('Review items:',len(issues))
print('Existing input files changed during collection:',len(changed))
print('No training, inference, statistical-test recomputation, R rerun, or figure generation was performed.')
print('\nUPLOAD THIS ZIP:',archive)
print('ZIP size: %.2f MiB'%(archive.stat().st_size/1024**2))
PYEOF
