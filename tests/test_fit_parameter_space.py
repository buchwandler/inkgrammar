import numpy as np
import pytest

from inkgrammar.fitting.models import PrimitiveTemplate
from inkgrammar.fitting.parameterization import ParameterSpace, ParameterSpec


def primitive():
    return PrimitiveTemplate.model_validate(
        {
            "id": "sas-underturn-01",
            "frame": {"start": [0.0, 0.0], "nominal_end": [1.0, 0.0]},
            "segments": [
                {"c1": [0.1, 0.4], "c2": [0.3, 0.8], "end": [0.5, 1.0]},
                {"c1": [0.7, 1.0], "c2": [0.9, 0.2], "end": [1.0, 0.0]},
            ],
            "free_parameters": ["segments.0.c1.x", "segments.0.c1.y"],
            "locked_parameters": ["frame.start", "segments.1.end"],
            "bounds": {
                "segments.0.c1.x": [-0.1, 0.3],
                "segments.0.c1.y": [0.1, 0.8],
            },
        }
    )


def test_parameter_space_encodes_and_decodes_typed_primitive_without_mutation():
    state = primitive()
    space = ParameterSpace.from_primitive(state)

    np.testing.assert_allclose(space.encode(state), [0.1, 0.4])
    fitted = space.decode([0.2, 0.6], state)

    assert isinstance(fitted, PrimitiveTemplate)
    assert fitted.segments[0].c1 == (0.2, 0.6)
    assert state.segments[0].c1 == (0.1, 0.4)
    assert space.bounds() == [(-0.1, 0.3), (0.1, 0.8)]


def test_parameter_space_rejects_wrong_shape_nonfinite_and_out_of_bounds_vectors():
    space = ParameterSpace.from_primitive(primitive())
    state = primitive()

    with pytest.raises(ValueError, match="shape"):
        space.decode([0.2], state)
    with pytest.raises(ValueError, match="finite"):
        space.decode([np.nan, 0.5], state)
    with pytest.raises(ValueError, match="outside bounds"):
        space.decode([0.31, 0.5], state)


def test_parameter_spec_and_space_reject_invalid_bounds_and_duplicate_names():
    with pytest.raises(ValueError, match="less than upper"):
        ParameterSpec("x", 1, 1, 1)
    with pytest.raises(ValueError, match="unique"):
        ParameterSpace([ParameterSpec("x", 0, 1, 0.5), ParameterSpec("x", 0, 1, 0.5)])
