"""ABC for bipartite graphs."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any

from geographer.graph.type_aliases import VertexColor, VertexName


class BipartiteGraphClass(ABC):
    """ABC for bipartite graphs.

    The decomposition of a vertex set into two sets are thought of as
    vertex colors, e.g. red or black.

    The graph can be directed or not and the edges can have extra data.
    e.g. a name, a color, a weight, possibly even composite data in the
    form of tuples.
    """

    def colors(self) -> list[VertexColor]:
        """Return vertex colors."""
        raise NotImplementedError

    def __opposite_color__(self, color: VertexColor) -> VertexColor:
        """Return opposite color."""
        raise NotImplementedError

    def vertices(self, color: VertexColor) -> list[VertexName]:
        """Return vertices of a given color."""
        raise NotImplementedError

    def vertices_opposite(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        edge_data: Any | None = None,
    ) -> list[VertexColor]:
        """Return list of adjacent vertices."""
        raise NotImplementedError

    def exists_vertex(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
    ) -> bool:
        """Return True if the vertex is in the graph, False otherwise."""
        raise NotImplementedError

    def exists_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Any | None,
    ) -> bool:
        """Return True if the edge is in the graph, False otherwise."""
        raise NotImplementedError

    def edge_data(
        self,
        from_vertex: VertexName,
        from_color: VertexColor,
        to_vertex: VertexName,
    ) -> Any:
        """Return edge data."""
        raise NotImplementedError

    def add_vertex(self, vertex_name: VertexName, vertex_color: VertexColor):
        """Add a vertex to the graph."""
        raise NotImplementedError

    def add_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Any,
        force: bool = False,
    ):
        """Add edge to graph."""
        raise NotImplementedError

    def delete_vertex(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        force_delete_with_edges=True,
    ):
        """Delete vertex from graph."""
        raise NotImplementedError

    def delete_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
    ):
        """Delete edge from graph."""
        raise NotImplementedError

    def save_to_file(self, file_path: Path | None = None):
        """Save graph to file."""
        raise NotImplementedError

    def really_undirected(self) -> bool:
        """Return True if graph is really undirected."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return str repr."""
        raise NotImplementedError
