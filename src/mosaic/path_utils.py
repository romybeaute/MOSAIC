from pathlib import Path
import os, yaml, warnings
from importlib.resources import files

# ---------- config loading ----------
def _default_cfg() -> dict:
    return yaml.safe_load((files("mosaic") / "configs" / "default.yaml").read_text())

def _user_cfg() -> dict:
    for p in [Path.home()/".config/mosaic/config.yaml",
              Path.home()/".mosaic/config.yaml"]:
        if p.exists():
            return yaml.safe_load(p.read_text())
    return {}

def load_cfg() -> dict:
    cfg = _default_cfg()
    cfg.update(_user_cfg())
    if "MOSAIC_DATA" in os.environ:
        cfg["data_root"] = os.environ["MOSAIC_DATA"]
    if "MOSAIC_BOX" in os.environ:
        cfg["box_root"] = os.environ["MOSAIC_BOX"]
    for k in ("data_root", "box_root"):
        if k in cfg and cfg[k]:
            cfg[k] = str(Path(os.path.expandvars(cfg[k])).expanduser())
    return cfg

CFG = load_cfg()

# ---------- helpers ----------
def _norm_dataset(name: str) -> str:
    return name.strip()

def data_root() -> Path:
    return Path(CFG["data_root"])

def box_root() -> Path:
    return Path(CFG["box_root"])

def raw_path(dataset: str, *parts: str) -> Path:
    """
    RAW (Box): <box_root>/<DATASET>/...
    Keeps dataset case (e.g., 'DREAMACHINE').
    """
    ds = _norm_dataset(dataset)
    return box_root().joinpath(ds, *parts)

def proc_path(dataset: str, *parts: str) -> Path:
    """
    PROCESSED: <data_root>/<dataset.lower()>/...
    """
    ds = _norm_dataset(dataset).lower()
    return data_root().joinpath(ds, *parts)

def eval_path(dataset: str, *parts: str) -> Path:
    """
    EVAL outputs: <data_root>/EVAL/<dataset.lower()>/...
    """
    ds = _norm_dataset(dataset).lower()
    return data_root().joinpath("EVAL", ds, *parts)

# ---------- (optional) backward-compat shims ----------
def raw_path_compat(dataset: str, _condition: str, *parts: str) -> Path:
    warnings.warn("raw_path(dataset, condition, ...) is deprecated; condition is ignored.", DeprecationWarning, stacklevel=2)
    return raw_path(dataset, *parts)

def proc_path_compat(dataset: str, _condition: str, *parts: str) -> Path:
    warnings.warn("proc_path(dataset, condition, ...) is deprecated; condition is ignored.", DeprecationWarning, stacklevel=2)
    return proc_path(dataset, *parts)



def project_root(start: Path | None = None) -> Path:
    """
    Return the repository root by walking upward until we find .git or pyproject.toml.
    This allows notebooks and scripts to locate repo-level folders consistently.
    """
    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()  # fallback
