import pytest
import numpy as np
import rustworkx as rx

@pytest.fixture
def hardware_graph():
    graph = rx.PyGraph()
    graph.add_nodes_from(["Q0", "Q1", "Q2"])
    graph.add_edges_from([
        (0, 1, 1.0),
        (1, 2, 1.0)
    ])
    
    return graph


def test_distance_matrix(hardware_graph):
    dist_matrix = rx.distance_matrix(hardware_graph)
    expected_matrix = np.array([
        [0., 1., 2.],
        [1., 0., 1.],
        [2., 1., 0.]
    ])
    
    np.testing.assert_array_equal(dist_matrix, expected_matrix)
