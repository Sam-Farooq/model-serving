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
