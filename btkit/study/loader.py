"""
Study YAML loader.

load_study() reads a study YAML, validates it into a StudyDefinition, and
returns both the model and the directory containing the YAML (needed by
StudyExpander to resolve relative strategy paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from btkit.study.definition import StudyDefinition


def load_study(path: str | Path) -> tuple[StudyDefinition, Path]:
    """
    Load and validate a study YAML file.

    Returns (StudyDefinition, study_dir) where study_dir is the directory
    containing the study YAML — used by StudyExpander to resolve strategy paths.

    Raises ValueError if the YAML lacks a top-level 'study' key.
    Raises ValidationError if the study structure is invalid.
    """
    path = Path(path)
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    study_data = raw.get("study")
    if study_data is None:
        raise ValueError(f"YAML file {path} must have a top-level 'study' key")

    definition = StudyDefinition.model_validate(study_data)
    return definition, path.parent
