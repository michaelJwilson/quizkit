import pytest
import numpy as np
import oxiquizkit

@pytest.fixture
def nb_large_inputs():
    shape = (1_000, 1_000)
    
    k = np.random.randint(0, 50, size=shape).astype(np.float64)
    n = np.random.randint(1, 10, size=shape).astype(np.float64)
    p = np.random.uniform(0.01, 0.99, size=shape).astype(np.float64)
    
    return k, n, p


def test_nb_performance(benchmark, nb_large_inputs):
    k, n, p = nb_large_inputs
    
    result = benchmark(oxiquizkit.nb, k, n, p)
    
    assert result.shape == k.shape
    assert result.dtype == np.float64
    
    assert not np.isnan(result).all()
