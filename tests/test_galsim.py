import pytest
import galsim
import numpy as np
from quizkit.simulation import get_galsim_image


@pytest.fixture
def galsim_image():
    return get_galsim_image()


def test_galsim_fixture(galsim_image):
    assert galsim_image is not None
