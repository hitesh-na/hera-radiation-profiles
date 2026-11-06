from pathlib import Path
import hashlib
import json
import platform
import re
import numpy as np


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import pandas, scipy, sklearn, xgboost, matplotlib
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "matplotlib": matplotlib.__version__}


def slug(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def new_output(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise ValueError(f"Output is not empty: {p}. Choose a NEW run directory.")
    p.mkdir(parents=True, exist_ok=True)
    return p
