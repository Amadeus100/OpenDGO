from opendgo_q1.topologies import FIXED_EDGES, diameter, sequential_inputs, sequential_layers


def test_paper_edge_sets_are_exact():
    assert FIXED_EDGES["star"] == ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
    assert FIXED_EDGES["chain"] == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
    assert FIXED_EDGES["tree"] == ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5))
    assert FIXED_EDGES["ring"] == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0))
    assert len(FIXED_EDGES["mesh"]) == 15
    assert FIXED_EDGES["community"] == (
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
        (2, 3),
    )


def test_sequential_dependency_is_a_leaf_to_sink_dag():
    assert sequential_layers(FIXED_EDGES["chain"], sink=0) == [[5], [4], [3], [2], [1]]
    assert sequential_inputs(FIXED_EDGES["chain"], sink=0)[3] == [4]
    assert sequential_layers(FIXED_EDGES["tree"], sink=0) == [[3, 4, 5], [1, 2]]
    assert diameter(FIXED_EDGES["chain"]) == 5
    assert diameter(FIXED_EDGES["mesh"]) == 1

