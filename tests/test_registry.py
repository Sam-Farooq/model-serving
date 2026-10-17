"""What the registry pins, against a real registry."""
from __future__ import annotations

import numpy as np

from serving.model import registry
from serving.schemas import FEATURE_ORDER


def test_the_version_is_a_string_as_the_response_schema_requires(register_model):
    # The registry returns the version as an int and it goes straight into
    # PredictResponse.model_version, which is typed as a string. Pydantic
    # refuses it, which turns every prediction into a 500 at the last line of
    # the handler.
    register_model(np.full(10, 0.5))

    _, version, feature_order = registry.load()

    assert version == "1"
    assert isinstance(version, str)
    assert feature_order == FEATURE_ORDER


def test_the_reference_is_read_back_off_the_pinned_run(register_model):
    register_model(np.random.default_rng(0).beta(2, 8, 2000).astype(np.float32))

    reference = registry.load_score_reference()

    assert reference is not None
    assert len(reference) == 2000
    assert all(0.0 <= score <= 1.0 for score in reference)


def test_the_reference_follows_the_pinned_version_not_the_newest_run(register_model):
    # Production stays on the first version. A later run logs a reference of
    # its own, and the monitor must not pick that one up, or a rollback
    # reports drift that is only the rollback.
    register_model(np.full(2000, 0.1, dtype=np.float32))
    register_model(np.full(1500, 0.9, dtype=np.float32), stage=None)

    reference = registry.load_score_reference()

    assert len(reference) == 2000
    assert max(reference) < 0.5


def test_a_run_with_no_reference_artifact_reads_back_as_none(register_model):
    register_model(None)

    assert registry.load_score_reference() is None


def test_the_reference_is_read_once_and_pinned(register_model):
    register_model(np.full(2000, 0.1, dtype=np.float32))

    first = registry.load_score_reference()
    second = registry.load_score_reference()

    assert second is first
