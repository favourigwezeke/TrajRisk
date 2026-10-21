"""Check portable source paths, notebook structure and recorded provenance."""
import ast
import hashlib
import json
from pathlib import Path
import re
from source_structure import science_hash

ROOT = Path(__file__).resolve().parents[1]
failures = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


changes = json.loads((ROOT / "metadata/public_source_changes.json").read_text())
moves = {row["previous_path"]: row["path"] for row in changes["moves"]}
if changes.get("executable_ast_hash_format") != "portable-python-ast-v1":
    raise SystemExit("Unsupported executable fingerprint format; verifier metadata must be updated together.")


def canonical_ast(value):
    """Serialize executable structure independently of ast.dump's Python version."""
    if isinstance(value, ast.AST):
        fields = {}
        for name, item in ast.iter_fields(value):
            # Python 3.12 adds this empty field to functions and classes.
            # Retain nonempty generic parameters, which are part of the code.
            if name == "type_params" and item == []:
                continue
            fields[name] = canonical_ast(item)
        return {"node": type(value).__name__, "fields": fields}
    if isinstance(value, list):
        return [canonical_ast(item) for item in value]
    if isinstance(value, bytes):
        return {"literal": "bytes", "hex": value.hex()}
    if isinstance(value, complex):
        return {"literal": "complex", "real": value.real, "imag": value.imag}
    if value is Ellipsis:
        return {"literal": "ellipsis"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported executable structure value: " + type(value).__name__)


def executable_hash(source):
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
    class Paths(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value
                for old, new in moves.items():
                    value = value.replace(new, old).replace(Path(new).name, Path(old).name)
                node.value = value
            return node
    representation = json.dumps(
        canonical_ast(Paths().visit(tree)), sort_keys=True, ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(representation.encode()).hexdigest()


for row in changes["moves"]:
    path = ROOT / row["path"]
    if row.get("archived_ref"):
        if path.exists() or (ROOT / row["previous_path"]).exists():
            failures.append("Archived preparation file remains in active tree: " + row["path"])
        continue
    if not path.is_file() or sha(path) != row["sha256"]:
        failures.append("Source rename checksum: " + row["path"])
    elif row.get("executable_ast_sha256") and executable_hash(path.read_text()) != row["executable_ast_sha256"]:
        failures.append("Executable code differs from original: " + row["path"])
    if (ROOT / row["previous_path"]).exists():
        failures.append("Retired active path remains: " + row["previous_path"])

provenance = json.loads((ROOT / "metadata/workflow_sources.json").read_text())
for row in provenance["files"]:
    relative = row.get("repository_path")
    if relative:
        path = ROOT / relative
        if not path.is_file() or sha(path) != row["portable_sha256"]:
            failures.append("Source provenance mismatch: " + relative)

code_cells = 0
notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
for path in notebooks:
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            code_cells += 1
            if cell.get("outputs") or cell.get("execution_count") is not None:
                failures.append("Notebook has retained execution output: " + path.name)
if len(notebooks) != 10 or code_cells != 97:
    failures.append("Preprocessing notebook/cell inventory differs from 10/97")

integrity = json.loads((ROOT / "metadata/source_integrity.json").read_text())
if integrity.get("format") != "scientific-python-ast-v1":
    failures.append("Unsupported scientific-source fingerprint format")
for row in integrity["files"]:
    path = ROOT / row["path"]
    if not path.is_file() or sha(path) != row["sha256"]:
        failures.append("Analysis source checksum mismatch: " + row["path"])
    elif science_hash(path.read_text()) != row["scientific_ast_sha256"]:
        failures.append("Scientific executable structure changed: " + row["path"])
for row in integrity["notebooks"]:
    notebook = json.loads((ROOT / row["path"]).read_text())
    hashes = [science_hash("".join(cell.get("source", []))) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if hashes != row["code_cell_hashes"]:
        failures.append("Notebook scientific structure changed: " + row["path"])

for path in (ROOT / "scripts").glob("*.py"):
    if re.search(r"week\d", path.name, re.I):
        failures.append("Development-stage active script name: " + path.name)
for folder in ["scripts", "notebooks", "docs"]:
    for path in (ROOT / folder).rglob("*"):
        if path.suffix in {".py", ".sh", ".R", ".md", ".ipynb"}:
            text = path.read_text()
            if re.search(r"/(?:Users|home)/[A-Za-z0-9_.-]+/", text):
                failures.append("Personal absolute directory: " + str(path.relative_to(ROOT)))
            if path.suffix == ".py":
                ast.parse(text, filename=str(path))

if failures:
    raise SystemExit("\n".join(failures))
print("Analysis source checksums and executable structure: PASS")
print("Preprocessing inventory: 10 notebooks / 97 cleared code cells")
print("Active source/document personal-directory scan: PASS")
