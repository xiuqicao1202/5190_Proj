"""
Local directory conventions:
- Pretrained_Params/: Pretrained weights (Hugging Face cache, glove.6B.100d.txt, etc.)
- Finetune_Params/: Fine-tuned artifacts after each run (timestamp prefix + like-named txt/py)
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.request import urlretrieve

GLOVE_ZIP_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_100D_TXT = "glove.6B.100d.txt"
# Reject tiny files (HTML error pages, empty files) before calling ZipFile.
MIN_GLOVE_ZIP_BYTES = 1_000_000


def _zip_members_include_glove_100d(zf: zipfile.ZipFile) -> bool:
    for name in zf.namelist():
        if name == GLOVE_100D_TXT or name.endswith("/" + GLOVE_100D_TXT):
            return True
    return False


def _glove_zip_is_valid(zip_path: Path) -> bool:
    """Return True only if the file looks like a complete Stanford GloVe zip."""
    try:
        if not zip_path.is_file():
            return False
        if zip_path.stat().st_size < MIN_GLOVE_ZIP_BYTES:
            return False
        if not zipfile.is_zipfile(zip_path):
            return False
        with zipfile.ZipFile(zip_path) as zf:
            if not _zip_members_include_glove_100d(zf):
                return False
    except (OSError, zipfile.BadZipFile):
        return False
    return True


def _download_glove_zip_atomic(zip_path: Path) -> None:
    """Download to a .part file then rename, so a failed run does not leave a falsely-named corrupt zip."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_suffix(zip_path.suffix + ".part")
    if partial.is_file():
        partial.unlink()
    print(
        f"[download] Fetching {GLOVE_ZIP_URL}\n         -> {zip_path} (large; may take several minutes)"
    )
    try:
        urlretrieve(GLOVE_ZIP_URL, str(partial))
        partial.replace(zip_path)
    except BaseException:
        if partial.is_file():
            partial.unlink()
        raise


def _extract_glove_100d_from_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(GLOVE_100D_TXT, dest_dir)


def project_root(explicit: Path | None, *, script_dir: Path) -> Path:
    return (explicit.expanduser().resolve() if explicit is not None else script_dir.resolve())


def pretrained_dir(project: Path) -> Path:
    d = project / "Pretrained_Params"
    d.mkdir(parents=True, exist_ok=True)
    return d


def finetune_base(project: Path) -> Path:
    d = project / "Finetune_Params"
    d.mkdir(parents=True, exist_ok=True)
    return d


def huggingface_pretrained_cache(project: Path) -> Path:
    d = pretrained_dir(project) / "huggingface"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_glove_txt(project: Path) -> Path:
    d = pretrained_dir(project) / "glove"
    d.mkdir(parents=True, exist_ok=True)
    return d / GLOVE_100D_TXT


def resolve_glove_txt(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return default_glove_txt(project)


def ensure_glove_100d(glove_txt: Path, *, auto_download: bool = True) -> None:
    """Use local glove.6B.100d.txt if present; otherwise download and extract the official zip (~862MB).

    Corrupt or HTML placeholder zips (common after failed downloads) are deleted and re-fetched.
    """
    if glove_txt.is_file():
        return
    if not auto_download:
        raise SystemExit(
            f"GloVe file not found: {glove_txt}\n"
            f"  Place {GLOVE_100D_TXT} in this directory, or remove --skip-glove-download "
            "to download glove.6B.zip automatically."
        )
    glove_txt.parent.mkdir(parents=True, exist_ok=True)
    zip_path = glove_txt.parent / "glove.6B.zip"

    if not _glove_zip_is_valid(zip_path):
        if zip_path.is_file():
            print(
                "[download] Removing invalid or incomplete glove.6B.zip "
                "(not a zip, truncated, or HTML error page)."
            )
            zip_path.unlink()
        _download_glove_zip_atomic(zip_path)

    if not _glove_zip_is_valid(zip_path):
        raise SystemExit(
            f"Could not obtain a valid glove.6B.zip after download.\n"
            f"  Delete {zip_path} if it exists, check network/VPN/firewall, then retry.\n"
            f"  Or manually place {GLOVE_100D_TXT} in: {glove_txt.parent}"
        )

    print(f"[download] Extracting {GLOVE_100D_TXT} …")
    try:
        _extract_glove_100d_from_zip(zip_path, glove_txt.parent)
    except zipfile.BadZipFile:
        if zip_path.is_file():
            print("[download] Zip failed at extract; removing and re-downloading once.")
            zip_path.unlink()
        _download_glove_zip_atomic(zip_path)
        if not _glove_zip_is_valid(zip_path):
            raise SystemExit(
                f"Re-downloaded zip is still invalid. Try manual download from:\n  {GLOVE_ZIP_URL}"
            )
        _extract_glove_100d_from_zip(zip_path, glove_txt.parent)

    if not glove_txt.is_file():
        raise SystemExit(f"After extraction, file still missing: {glove_txt}")


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def finetune_prefix(script_stem: str) -> str:
    return f"{timestamp_run_id()}_{script_stem}"


def copy_script_source(script_file: Path, dest_py: Path) -> None:
    shutil.copy2(script_file, dest_py)


def write_training_txt(path: Path, sections: Mapping[str, Any]) -> None:
    out: list[str] = []
    for title, body in sections.items():
        out.append(f"======== {title} ========")
        if isinstance(body, (dict, list)):
            out.append(json.dumps(body, indent=2, ensure_ascii=False))
        else:
            out.append(str(body))
        out.append("")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
