"""Print and snapshot the exact source used by an experiment."""

import hashlib
import json
from pathlib import Path
import shutil


def snapshot_sources(project_root: Path, destination: Path) -> dict:
    """Save all implementation modules and dependency files before training."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    paths = sorted((project_root / "src").rglob("*.py"))
    paths += [
        project_root / name
        for name in ("pyproject.toml", "uv.lock")
        if (project_root / name).is_file()
    ]
    for path in paths:
        relative = path.relative_to(project_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest[str(relative)] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
        if path.suffix == ".py":
            print(
                f"SOURCE {relative}\n{target.read_text(encoding='utf-8')}",
                flush=True,
            )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
