class BipartiteGraphClass:
    """Abstract class for bipartite graphs (the decomposition of a vertex set
    into two sets are thought of as vertex colors, e.g. red or black) which can
    be serialized in/read in from a json file.

    The graph can be directed or not and the edges can have extra data.
    e.g. a name, a color, a weight, possibly even composite data in the
    form of tuples.
    """

    def colors(self):
        raise NotImplementedError

    def __opposite_color__(self, color):
        raise NotImplementedError

    def vertices(self, color):
        raise NotImplementedError

    def vertices_opposite(self, vertex, vertex_color, edge_data=None):
        raise NotImplementedError

    def exists_vertex(self, vertex_name, vertex_color):
        raise NotImplementedError

    def exists_edge(self, from_vertex, from_vertex_color, to_vertex):
        raise NotImplementedError

    def edge_data(from_vertex, from_color, to_vertex):
        raise NotImplementedError

    def add_vertex(self, vertex_name, vertex_color):
        raise NotImplementedError

    def add_edge(self,
                 from_vertex,
                 from_vertex_color,
                 to_vertex,
                 edge_data,
                 force=False):
        raise NotImplementedError

    def delete_vertex(self,
                      vertex,
                      vertex_color,
                      force_delete_with_edges=True):
        raise NotImplementedError

    def delete_edge(self, from_vertex, from_vertex_color, to_vertex):
        raise NotImplementedError

    def save_to_file(self, file_path=None):
        raise NotImplementedError

    def really_undirected(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
