from pathlib import Path
from typing import Any, Optional

from geographer.graph.type_aliases import VertexColor, VertexName


class BipartiteGraphClass:
    """Abstract class for bipartite graphs (the decomposition of a vertex set
    into two sets are thought of as vertex colors, e.g. red or black) which can
    be serialized in/read in from a json file.

    The graph can be directed or not and the edges can have extra data.
    e.g. a name, a color, a weight, possibly even composite data in the
    form of tuples.
    """

    def colors(self) -> list[VertexColor]:
        raise NotImplementedError

    def __opposite_color__(self, color: VertexColor) -> VertexColor:
        raise NotImplementedError

    def vertices(self, color: VertexColor) -> list[VertexName]:
        raise NotImplementedError

    def vertices_opposite(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        edge_data: Optional[Any] = None,
    ) -> list[VertexColor]:
        raise NotImplementedError

    def exists_vertex(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
    ) -> bool:
        raise NotImplementedError

    def exists_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Optional[Any],
    ) -> bool:
        raise NotImplementedError

    def edge_data(
        self,
        from_vertex: VertexName,
        from_color: VertexColor,
        to_vertex: VertexName,
    ) -> Any:
        raise NotImplementedError

    def add_vertex(self, vertex_name: VertexName, vertex_color: VertexColor):
        raise NotImplementedError

    def add_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Any,
        force: bool = False,
    ):
        raise NotImplementedError

    def delete_vertex(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        force_delete_with_edges=True,
    ):
        raise NotImplementedError

    def delete_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
    ):
        raise NotImplementedError

    def save_to_file(self, file_path: Optional[Path] = None):
        raise NotImplementedError

    def really_undirected(self) -> bool:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError
