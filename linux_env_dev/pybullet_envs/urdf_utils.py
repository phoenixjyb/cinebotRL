import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict


def rewrite_urdf_text(urdf_text: str, package_rewrites: Dict[str, str]) -> str:
    out = urdf_text
    for src, dst in (package_rewrites or {}).items():
        out = out.replace(str(src), str(dst))
    return out


def prepare_urdf_for_pybullet(urdf_path: str, package_rewrites: Dict[str, str], cache_dir: str) -> str:
    """Return a URDF path that PyBullet can load.

    If `package_rewrites` is non-empty, we rewrite `package://...` references and
    cache the rewritten URDF under `cache_dir`.
    """
    if urdf_path is None or str(urdf_path).strip() == "":
        raise ValueError("urdf_path is empty")

    urdf_path = os.path.abspath(str(urdf_path))
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    if not package_rewrites:
        return urdf_path

    with open(urdf_path, "rb") as f:
        urdf_bytes = f.read()
    try:
        urdf_text = urdf_bytes.decode("utf-8")
    except Exception:
        urdf_text = urdf_bytes.decode("utf-8", errors="replace")

    rewritten = rewrite_urdf_text(urdf_text, package_rewrites)
    if rewritten == urdf_text:
        return urdf_path

    cache_dir = os.path.abspath(str(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    # Content-addressed cache key (includes rewrite mapping).
    h = hashlib.sha256()
    h.update(urdf_bytes)
    h.update(json.dumps(package_rewrites, sort_keys=True).encode("utf-8"))
    digest = h.hexdigest()[:12]
    out_path = Path(cache_dir) / f"{Path(urdf_path).stem}.{digest}.urdf"
    if out_path.is_file():
        return str(out_path)

    # Atomic write to avoid races under SubprocVecEnv workers.
    fd, tmp_path = tempfile.mkstemp(prefix=out_path.stem + ".", suffix=".tmp", dir=cache_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(rewritten)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    return str(out_path)

