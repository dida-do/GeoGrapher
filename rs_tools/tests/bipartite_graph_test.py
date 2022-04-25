"""Pytest test suite for BipartiteGraph class."""

import json

import pytest

from rs_tools.graph.bipartite_graph import empty_bipartite_graph


def test_bipartite_graph():
    # test adding vertices
    graph1 = empty_bipartite_graph()

    red_vertices = set({'r1', 'r2', 'r3'})
    for vertex in red_vertices:
        graph1.add_vertex(vertex, 'red')

    black_vertices = set({'b1', 'b2', 'b3'})
    for vertex in black_vertices:
        graph1.add_vertex(vertex, 'black')

    # test internal graph_dict is as it should be
    assert graph1._graph_dict == {
        'red': {
            'r3': {},
            'r1': {},
            'r2': {}
        },
        'black': {
            'b3': {},
            'b2': {},
            'b1': {}
        }
    }

    # test exists_vertex on existing vertices
    for vertex in red_vertices:
        assert graph1.exists_vertex(vertex, 'red') == True

    for vertex in black_vertices:
        assert graph1.exists_vertex(vertex, 'black') == True

    # test exists_vertex on non-existing vertices
    non_existing_red_vertices = set({'not_r1', 'not_r2', 'not_r3'})
    for vertex in non_existing_red_vertices:
        assert graph1.exists_vertex(vertex, 'red') == False

    non_existing_black_vertices = set({'not_b1', 'not_b2', 'not_b3'})
    for vertex in non_existing_black_vertices:
        assert graph1.exists_vertex(vertex, 'black') == False

    # test adding not yet existing edges with existing vertices
    edges_with_existing_vertices = set({
        ('r1', 'red', 'b1', "foo"),
        ('r1', 'red', 'b2', "foo"),
        ('r3', 'red', 'b1', "bar"),
    })
    for from_vertex, from_color, to_vertex, edge_data in edges_with_existing_vertices:
        graph1.add_edge(from_vertex, from_color, to_vertex, edge_data)

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "bar"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo"
            }
        },
        'black': {
            'b3': {},
            'b2': {
                'r1': "foo"
            },
            'b1': {
                'r1': "foo",
                'r3': "bar"
            }
        }
    }

    # test adding existing edges with force=False
    with pytest.raises(Exception):
        graph1.add_edge('r1', 'red', 'b1',
                        "foo")  # edge was in edges_with_existing_vertices

    # test adding edges with only one existing vertex which is of the same color as the source vertex
    edges_with_one_existing_vertices_of_source_color = set({
        ('r1', 'red', 'b4', "foo"),
        ('r1', 'red', 'b5', "foo"),
        ('r3', 'red', 'b6', "bar"),
    })
    for from_vertex, from_color, to_vertex, edge_data in edges_with_one_existing_vertices_of_source_color:
        graph1.add_edge(from_vertex, from_color, to_vertex, edge_data)

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "bar",
                'b6': "bar"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo",
                'b4': "foo",
                'b5': "foo"
            }
        },
        'black': {
            'b3': {},
            'b1': {
                'r1': "foo",
                'r3': "bar"
            },
            'b2': {
                'r1': "foo"
            },
            'b4': {
                'r1': "foo"
            },
            'b6': {
                'r3': "bar"
            },
            'b5': {
                'r1': "foo"
            }
        }
    }

    # test adding edges with only one existing vertex which is of the opposite color as the source vertex
    edges_with_one_existing_vertices_of_opposite_color = set({
        ('r4', 'red', 'b4', "foo"),
        ('r5', 'red', 'b4', "foo"),
    })
    for from_vertex, from_color, to_vertex, edge_data in edges_with_one_existing_vertices_of_opposite_color:
        graph1.add_edge(from_vertex, from_color, to_vertex, edge_data)

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "bar",
                'b6': "bar"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo",
                'b4': "foo",
                'b5': "foo"
            },
            'r4': {
                'b4': "foo"
            },
            'r5': {
                'b4': "foo"
            }
        },
        'black': {
            'b3': {},
            'b1': {
                'r1': "foo",
                'r3': "bar"
            },
            'b2': {
                'r1': "foo"
            },
            'b4': {
                'r1': "foo",
                'r4': "foo",
                'r5': "foo"
            },
            'b6': {
                'r3': "bar"
            },
            'b5': {
                'r1': "foo"
            }
        }
    }

    # test adding edges without existing vertices
    edges_without_existing_vertices = set({
        ('r6', 'red', 'b7', "foo"),
        ('r7', 'red', 'b8', "bar"),
    })
    for from_vertex, from_color, to_vertex, edge_data in edges_without_existing_vertices:
        graph1.add_edge(from_vertex, from_color, to_vertex, edge_data)

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "bar",
                'b6': "bar"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo",
                'b4': "foo",
                'b5': "foo"
            },
            'r4': {
                'b4': "foo"
            },
            'r5': {
                'b4': "foo"
            },
            'r7': {
                'b8': "bar"
            },
            'r6': {
                'b7': "foo"
            }
        },
        'black': {
            'b3': {},
            'b1': {
                'r1': "foo",
                'r3': "bar"
            },
            'b2': {
                'r1': "foo"
            },
            'b4': {
                'r1': "foo",
                'r4': "foo",
                'r5': "foo"
            },
            'b6': {
                'r3': "bar"
            },
            'b5': {
                'r1': "foo"
            },
            'b8': {
                'r7': "bar"
            },
            'b7': {
                'r6': "foo"
            }
        }
    }

    # test adding edges with existing edges, force = True
    edges_with_existing_vertices = set({
        ('r1', 'red', 'b1', "foo"),
        ('r1', 'red', 'b2', "foo"),
        ('r3', 'red', 'b1', "foo"),  # existing edge with new edge_data
    })
    for from_vertex, from_color, to_vertex, edge_data in edges_with_existing_vertices:
        graph1.add_edge(from_vertex,
                        from_color,
                        to_vertex,
                        edge_data,
                        force=True)

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "foo",
                'b6': "bar"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo",
                'b4': "foo",
                'b5': "foo"
            },
            'r4': {
                'b4': "foo"
            },
            'r5': {
                'b4': "foo"
            },
            'r7': {
                'b8': "bar"
            },
            'r6': {
                'b7': "foo"
            }
        },
        'black': {
            'b3': {},
            'b1': {
                'r1': "foo",
                'r3': "foo"
            },
            'b2': {
                'r1': "foo"
            },
            'b4': {
                'r1': "foo",
                'r4': "foo",
                'r5': "foo"
            },
            'b6': {
                'r3': "bar"
            },
            'b5': {
                'r1': "foo"
            },
            'b8': {
                'r7': "bar"
            },
            'b7': {
                'r6': "foo"
            }
        }
    }

    # since we have a nice, full graph now, let's test vertices_opposite
    assert set(graph1.vertices_opposite('b4',
                                        'black')) == set({'r1', 'r4', 'r5'})
    # test with filtered edge_data: replace edge_data on edge b4 -> r1 to "bar"
    graph1.add_edge('b4', 'black', 'r1', "bar", force=True)
    assert set(graph1.vertices_opposite('b4', 'black',
                                        edge_data="foo")) == set({'r4', 'r5'})
    assert set(graph1.vertices_opposite('b4', 'black',
                                        edge_data="bar")) == set({'r1'})

    # test deleting existing vertices with force_delete_with_edges set to "foo"
    red_vertices_to_delete = set({'r4', 'r5', 'r6', 'r7'})
    for vertex in red_vertices_to_delete:
        graph1.delete_vertex(vertex, 'red')

    black_vertices_to_delete = set({'b4', 'b5', 'b6', 'b7', 'b8'})
    for vertex in black_vertices_to_delete:
        graph1.delete_vertex(vertex, 'black')

    assert graph1._graph_dict == {
        'red': {
            'r3': {
                'b1': "foo"
            },
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo"
            }
        },
        'black': {
            'b3': {},
            'b2': {
                'r1': "foo"
            },
            'b1': {
                'r1': "foo",
                'r3': "foo"
            }
        }
    }

    # test deleting non-existing vertices
    graph1.delete_vertex('r12341234', 'red')

    # test deleting existing vertices with force_delete_with_edges set to False
    # vertex with no edges:
    graph1.add_vertex('r1000', 'red')
    graph1.delete_vertex('r1000', 'red', force_delete_with_edges=False)
    # vertex with edges:
    with pytest.raises(Exception):
        graph1.delete_vertex('r1', 'red', force_delete_with_edges=False)

    # test deleting edges
    graph1.delete_edge('r3', 'red', 'b1')

    assert graph1._graph_dict == {
        'red': {
            'r3': {},
            'r2': {},
            'r1': {
                'b1': "foo",
                'b2': "foo"
            }
        },
        'black': {
            'b3': {},
            'b2': {
                'r1': "foo"
            },
            'b1': {
                'r1': "foo"
            }
        }
    }

    # test really_undirected
    assert graph1.really_undirected() == True


# TODO: remove once I get pytest to run
if __name__ == "__main__":
    test_bipartite_graph()