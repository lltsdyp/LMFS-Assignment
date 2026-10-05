"""Owns W&B experiment setup and source-code artifact recording."""

from collections.abc import Iterator
import contextlib
import pathlib

import wandb

from src.utils import source


@contextlib.contextmanager
def experiment(
    task: str,
    config: dict[str, object],
    *,
    project: str,
    entity: str | None = None,
    mode: str | None = None,
) -> Iterator[wandb.Run]:
    """Creates and finishes a configured W&B experiment.

    Args:
        task: Task name used as the run job type.
        config: Complete model and corpus configuration.
        project: W&B project name.
        entity: Optional W&B user or team.
        mode: Online, offline, disabled, or None to respect W&B settings.

    Yields:
        The active run, finished automatically when the context exits.
    """
    with wandb.init(
        project=project, entity=entity, job_type=task, config=config, mode=mode
    ) as run:
        yield run


def log_sources(run: wandb.Run, project_root: pathlib.Path) -> None:
    """Records the CLI and all implementation modules as a code artifact."""
    artifact = wandb.Artifact(f"bpe-source-{run.id}", type="code")
    for path in source.implementation_paths(project_root):
        artifact.add_file(str(path), name=str(path.relative_to(project_root)))
    run.log_artifact(artifact)
