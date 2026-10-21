#!/usr/bin/env bash
# Replay the author-supplied frozen packet; writes only beneath build/.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  printf 'Usage: bash scripts/reproduce_saved.sh /path/to/TrajRisk_frozen_inputs.zip\n' >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
packet_argument="$1"
python3 "$repo_root/scripts/verify_repository.py"
python3 "$repo_root/scripts/verify_release_assets.py"
python3 "$repo_root/scripts/reproduce_frozen.py" --bundle "$packet_argument" --bootstrap-ci
packet_root="$(python3 - "$packet_argument" "$repo_root/build/frozen_replay/inputs" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).expanduser().resolve()
if p.is_file():
    p = Path(sys.argv[2]).resolve()
if not (p / 'PACKET_MANIFEST.json').is_file():
    matches = list(p.glob('*/PACKET_MANIFEST.json'))
    if len(matches) != 1:
        raise SystemExit('Cannot resolve the verified input-packet directory')
    p = matches[0].parent
print(p)
PY
)"
export TRAJRISK_BUNDLE="$packet_root"
export TRAJRISK_ROOT="$packet_root/originals"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for number in 1 2 3 4 5; do
  python3 "$repo_root/scripts/figures/make_figure_s${number}.py"
done
printf '\nSaved-input replay complete. Outputs are under build/frozen_replay/ and build/figures/supplementary/.\n'
