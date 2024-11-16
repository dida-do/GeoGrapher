"""Simple class for bipartite graphs.

Implements graphs adjacency list-style as (dict-of)-dict-of-dicts.

Example:
A graph with

* vertex colors 'red' and 'black'
* red vertices 'r1', 'r2' and black vertices 'b1', 'b2', 'b3'
* an edge with edge data 'foo' between 'r1' and 'b1' and  an edge with edge data
  'bar' between 'r1' and 'b3'

is encoded as the dict of dicts of dicts::

    {
        'red':
            {
                'r1': {'b1': 'foo', 'b3': 'bar'},
                'r2': {},
                'r3': {}
            }

        'black':
            {
                'b1': {'r1': 'foo'},
                'b2': {},
                'b3': {'r1': 'bar'}
            }
    }


Only tested for the undirected case.
"""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from geographer.graph.bipartite_graph_class import BipartiteGraphClass
from geographer.graph.type_aliases import VertexColor, VertexName

# logger
log = logging.Logger(__name__)


def empty_graph_dict(red="red", black="black"):
    """Return graph dict of empty bipartite graph.

    Args:
        red: vertex color. Defaults to "red".
        black: vertex color. Defaults to "black".

    Returns:
        graph dict of empty graph
    """
    return {red: {}, black: {}}


def empty_bipartite_graph(red="red", black="black"):
    """Return empty bipartite graph.

    Args:
        red: vertex color. Defaults to "red".
        black: vertex color. Defaults to "black".

    Returns:
        empty graph
    """
    return BipartiteGraph(graph_dict=empty_graph_dict(red=red, black=black))


class BipartiteGraph(BipartiteGraphClass):
    """Class implementing bipartite graphs.

    Simple adjacency list-style dict-of-dicts (sort of) implementation
    of a bipartite graph (a vertex is thought of as being either red or
    black) which can be serialized in/read in from a json file. The
    graph can be directed or not and the edges can have extra data. e.g.
    a name, a color, a weight, possibly even composite data in the form
    of tuples, though I haven't checked if that plays well with json.
    The vertices have to be hashable, e.g. strings.
    """

    def __init__(
        self,
        graph_dict: dict | None = None,
        file_path: Path | None = None,
        red: VertexColor = None,
        black: VertexColor = None,
        directed: bool = False,
    ):
        """Construct graph from graph dict or json file.

        Construct from either a dict (of dicts of dicts) or from
        a json file serializing such a dict. The encoding is such that
        graph_dict[vertex1_color][vertex1][vertex2] is the edge data between
        vertex1 (of a certain color vertex1_color) and vertex2 (of the opposite
        color). A vertex v of color c without an edge is encoded by having
        graph_dict[c][v] be the empty dict. A directed graph is encoded as an
        undirected graph with edges in both directions.

        Args:
            graph_dict: dict (of dicts of dicts) defining a bipartite graph.
                See example in module header.
            file_path:path to .json containing such a dict.
            red: vertex color, defaults to 'red'.
            black: vertex color, defaults to 'black.
            directed: If True the graph is directed, defaults to False.
        """
        if file_path is not None:
            self.file_path: Path | None = file_path
            self.directed = directed
            try:
                with open(file_path, "r") as file:
                    self._graph_dict = json.load(file)
            except FileNotFoundError:
                log.exception("Graph dict file %s not found", file_path)
            except JSONDecodeError:
                log.exception("Json file %s could not be decoded.", file_path)

            if len(self._graph_dict) != 2:
                raise ValueError(
                    "__init__: input argument graph_dict must have outer dict be "
                    "of length two!"
                )
            # vertex colors
            self.red, self.black = tuple(self._graph_dict.keys())

        elif graph_dict is not None:  # color determined by graph dict
            # make sure outermost dict has right shape
            if len(graph_dict) != 2:
                raise ValueError(
                    "__init__: input argument graph_dict must have "
                    "outer dict be of length two!"
                )
            self._graph_dict = graph_dict
            self.directed = directed
            # vertex colors
            self.red, self.black = tuple(graph_dict.keys())
        elif red is not None or black is not None:
            if red is not None and black is not None:
                self.red = red
                self.black = black
            else:
                raise Exception(
                    "Error: Need either both or none of red and black specified!"
                )
        else:
            self.red = "red"
            self.black = "black"
            self._graph_dict = empty_graph_dict(red=self.red, black=self.black)
            self.directed = directed
            self.file_path = None

    def colors(self) -> list[VertexColor]:
        """Return vertex colors."""
        return [self.red, self.black]

    def _opposite_color(self, color: VertexColor) -> VertexColor:
        """Return opposite vertex color."""
        if color == self.red:
            answer = self.black
        elif color == self.black:
            answer = self.red
        else:
            raise Exception(f"not a valid color: {color}")
        return answer

    def vertices(self, color: VertexColor) -> list[VertexName]:
        """Return vertices of a given color."""
        return self._graph_dict[color].keys()

    def vertices_opposite(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        edge_data: Any | None = None,
    ) -> list[VertexColor]:
        """Return list of adjacent vertices.

        Since our graph is bipartite, these are always of the opposite
        color, hence 'opposite'.
        """
        answer = self._graph_dict[vertex_color][vertex_name].keys()
        if edge_data is not None:
            answer = map(
                lambda opp_vertex_edge_data_pair: opp_vertex_edge_data_pair[0],
                filter(
                    lambda opp_vertex_edge_data_pair: opp_vertex_edge_data_pair[1]
                    == edge_data,
                    self._graph_dict[vertex_color][vertex_name].items(),
                ),
            )
        return list(answer)

    def exists_vertex(self, vertex_name: VertexName, vertex_color: VertexColor) -> bool:
        """Return True if the vertex is in the graph, False otherwise."""
        # whether vertex exists in either color
        exists_color_red = vertex_name in self._graph_dict[self.red]
        exists_color_black = vertex_name in self._graph_dict[self.black]

        if vertex_color is None:
            answer = exists_color_red or exists_color_black
        elif vertex_color == self.red:
            answer = exists_color_red
        elif vertex_color == self.black:
            answer = exists_color_black
        else:
            log.error(
                "Not a valid vertex_color: %s. Graph vertex colors are %s.",
                vertex_color,
                self.colors(),
            )

        return answer

    def exists_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Any | None = None,
    ) -> bool:
        """Return True if the edge is in the graph, False otherwise."""
        if edge_data is None:
            answer = to_vertex in self._graph_dict[from_vertex_color][from_vertex]
        else:
            answer = to_vertex in self.vertices_opposite(
                from_vertex, from_vertex_color, edge_data=edge_data
            )
        return answer

    def edge_data(
        self, from_vertex: VertexName, from_color: VertexColor, to_vertex: VertexName
    ) -> Any:
        """Return edge data.

        Args:
            from_vertex:
            from_color:
            to_vertex:
        """
        return self._graph_dict[from_color][from_vertex][to_vertex]

    def add_vertex(self, vertex_name: VertexName, vertex_color: VertexColor):
        """Add a vertex to the graph.

        Args:
            vertex_name
            vertex_color
        """
        # check if vertex already exists
        if self.exists_vertex(vertex_name, vertex_color=vertex_color):
            log.info("Vertex %s of already exists!", vertex_name)
        else:
            # create vertex w/o edges
            self._graph_dict[vertex_color][vertex_name] = {}

    def add_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
        edge_data: Any,
        force: bool = False,
    ):
        """Add an edge to the graph.

        If the vertices do not yet exist will create them.
        Throws an error if an edge between the vertices already exists unless
        force is True, in which case it overwrites the existing edge_data.

        Args:
            from_vertex:
            from_color:
            to_vertex:
            edge_data:
            force:
        """
        if not self.exists_vertex(from_vertex, from_vertex_color):
            log.info(
                "add_edge: vertex %s does not exist. Creating first...", from_vertex
            )
            self.add_vertex(from_vertex, from_vertex_color)
            # self.add_edge(from_vertex, from_vertex_color, to_vertex, edge_data, force)
        if not self.exists_vertex(to_vertex, self._opposite_color(from_vertex_color)):
            log.info("add_edge: vertex %s does not exist. Creating first...", to_vertex)
            self.add_vertex(to_vertex, self._opposite_color(from_vertex_color))
            # self.add_edge(from_vertex, from_vertex_color, to_vertex, edge_data, force)
        if (not force) and self.exists_edge(from_vertex, from_vertex_color, to_vertex):
            log.error(
                "add_edge: an edge %s (color: %s) to %s already exists. "
                "Set force=True to overwrite.",
                from_vertex,
                from_vertex_color,
                to_vertex,
            )
            raise Exception(
                f"add_edge: an edge {from_vertex} (color: {from_vertex_color}) "
                f"to {to_vertex} already exists. Set force=True to overwrite."
            )
        else:
            # add edge or update edge_date
            self._graph_dict[from_vertex_color][from_vertex][to_vertex] = edge_data

            # add or update opposite edge if graph is undirected:
            if not self.directed:
                to_vertex_color = self._opposite_color(from_vertex_color)
                self._graph_dict[to_vertex_color][to_vertex][from_vertex] = edge_data

    def delete_vertex(
        self,
        vertex_name: VertexName,
        vertex_color: VertexColor,
        force_delete_with_edges=True,
    ):
        """Delete a vertex from the graph.

        If force==False, will delete only if the vertex has
        no edges. If force==True, will also delete edges starting or ending at
        the vertex. Note that in that case we don't delete dangling opposite
        vertices (i.e. opposite vertices that don't have any edges left after
        this). Only implemented for directed graphs.

        Args:
            vertex_name:
            vertex_color:
            force_delete_with_edges:
        """
        if not self.exists_vertex(vertex_name, vertex_color):
            log.info(
                "delete_vertex: nothing to do, vertex %s does not exist.", vertex_name
            )

        # if force_delete_with_edges=False check if vertex has outgoing adjacent edges
        elif self.directed:
            log.error(
                "Sorry, delete_vertex is not implemented for directed graphs. "
                "I was too lazy to code up the complication of checking "
                "which edges end in %s :(",
                vertex_name,
            )

            raise Exception(
                f"Sorry, delete_vertex is not implemented for directed graphs. I was "
                "too lazy to code up the complication of checking which edges end in "
                f"{vertex_name} :("
            )

        elif (
            not force_delete_with_edges
            and list(self.vertices_opposite(vertex_name, vertex_color)) != []
        ):
            raise Exception(
                f"delete_vertex: vertex {vertex_name} of color {vertex_color} has "
                "edges. Set force_delete_with_edges=True to delete anyway "
                "(along with adjacent edges)."
            )

        else:
            # thinking of an undirected graph as a directed graph where for each edge
            # there is an opposite edge, we first take out the edges _ending_ in
            # vertex, i.e. the opposite edges to the outgoing ones at vertex.
            opposite_color = self._opposite_color(vertex_color)
            for opposite_vertex in self._graph_dict[vertex_color][vertex_name].keys():
                self._graph_dict[opposite_color][opposite_vertex].pop(vertex_name)
            # then we take out the edges starting in vertex and the vertex itself
            self._graph_dict[vertex_color].pop(vertex_name)

    def delete_edge(
        self,
        from_vertex: VertexName,
        from_vertex_color: VertexColor,
        to_vertex: VertexName,
    ):
        """Delete an edge from the graph.

        Delete the edge between vertices from_vertex of color
        from_vertex_color and to_vertex.

        Args:
            from_vertex:
            from_vertex_color:
            to_vertex:
        """
        if to_vertex not in self._graph_dict[from_vertex_color][from_vertex]:
            log.info(
                "delete_edge(%s, %s, %s): There is no such edge.",
                from_vertex,
                from_vertex_color,
                to_vertex,
            )
        else:
            self._graph_dict[from_vertex_color][from_vertex].pop(to_vertex)
            if not self.directed:  # delete opposite edge
                opposite_color = self._opposite_color(from_vertex_color)
                self._graph_dict[opposite_color][to_vertex].pop(from_vertex)

    def save_to_file(self, file_path: Path | None = None):
        """Save graph (i.e. graph_dict) to disk as json file.

        Args:
            file_path: path of json file to save graph to.
        """
        if file_path is None:
            if self.file_path is not None:
                file_path = self.file_path
            else:
                raise Exception(
                    "save_to_file: no file_path on record, "
                    "specify as file_path argument."
                )
        else:
            self.file_path = file_path
            with open(file_path, "w", encoding="utf-8") as write_file:
                json.dump(self._graph_dict, write_file, indent=4, ensure_ascii=False)

    def really_undirected(self) -> bool:
        """Check if graph is really undirected.

        Check if graph is really undirected, i.e. if for each edge the
        opposite edge exists as well. Useful for testing.

        Returns:
            True if graph is undirected, False if it's not.
        """
        # an empty graph is undirected
        answer = True

        try:
            for color in self._graph_dict.keys():
                opposite_color = self._opposite_color(color)
                for vertex in self._graph_dict[color]:
                    for opposite_vertex in self._graph_dict[color][vertex]:
                        if (
                            self._graph_dict[color][vertex][opposite_vertex]
                            != self._graph_dict[opposite_color][opposite_vertex][vertex]
                        ):
                            answer = False
        except KeyError:
            answer = False

        return answer

    def __eq__(self, other) -> bool:
        """Check equality of graphs.

        Two graphs are equal if the vertex sets
        and colors and the edge sets and edge data agree, which is tested by
        asking whether the underlying dicts are equal.

        Args:
            other (BipartiteGraph):

        Returns:
            True if the graphs are equal, False otherwise.
        """
        return self._graph_dict == other._graph_dict

    def __str__(self):
        """Return string representation of the graph.

        Args:
        Returns:
            (str): string representation of the graph.
        """
        return json.dumps(self._graph_dict, indent=4)
