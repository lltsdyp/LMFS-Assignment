"""Print and snapshot the exact source used by an experiment."""

import hashlib
import json
from pathlib import Path
import shutil


def implementation_paths(project_root: Path) -> list[Path]:
    """Lists Python/Rust implementation and native build configuration files."""
    paths = sorted((project_root / "src").rglob("*.py"))
    paths += sorted((project_root / "rust" / "src").rglob("*.rs"))
    paths += [
        project_root / "rust" / name
        for name in ("pyproject.toml", "Cargo.toml", "Cargo.lock")
        if (project_root / "rust" / name).is_file()
    ]
    return paths


def snapshot_sources(project_root: Path, destination: Path) -> dict:
    """Save all implementation modules and dependency files before training."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    paths = implementation_paths(project_root)
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
        if path.suffix in {".py", ".rs"}:
            print(
                f"SOURCE {relative}\n{target.read_text(encoding='utf-8')}",
                flush=True,
            )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
