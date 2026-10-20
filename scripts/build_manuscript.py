"""Compile current active sources without altering the supplied inputs."""
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
compiler = shutil.which('pdflatex')
if compiler is None:
    raise SystemExit('pdfLaTeX is required. Alternatively, compile manuscript/review_all.tex in Overleaf.')
build = root / 'build'
build.mkdir(exist_ok=True)
for pass_number in range(1, 4):
    result = subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error',
                             '-output-directory=' + str(build), 'review_all.tex'],
                            cwd=root/'manuscript', capture_output=True, text=True)
    (build / ('compile_pass_' + str(pass_number) + '.txt')).write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout[-5000:], file=sys.stderr)
        raise SystemExit(result.returncode)
output = build/'TrajRisk_Combined_Review.pdf'
shutil.copyfile(build/'review_all.pdf', output)
print(output)
