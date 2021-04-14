"""
Simple class for bipartite graphs, implemented adjacency list-style as (dict-of)-dict-of-dicts. 

Example: 
A graph with 
    * vertex colors 'red' and 'black'
    * red vertices 'r1', 'r2' and black vertices 'b1', 'b2', 'b3'
    * an edge with edge data 'foo' between 'r1' and 'b1' and  an edge with edge data 'bar' between 'r1' and 'b3' 
is encoded as the dict of dicts of dicts

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

import logging
import json
from json import JSONDecodeError
from rs_tools.graph.bipartite_graph_class import BipartiteGraphClass

# logger 
log = logging.Logger(__name__)


def empty_graph_dict(red='red', black='black'):
    return {red:{}, black:{}} 


def empty_bipartite_graph(red='red', black='black'):
    return BipartiteGraph(graph_dict=empty_graph_dict(red=red, black=black))


class BipartiteGraph(BipartiteGraphClass):
    """
    Class implementing bipartite graphs.

    Simple adjacency list-style dict-of-dicts (sort of) implementation of a bipartite graph  
    (a vertex is thought of as being either red or black) which can be serialized in/read in from a json file. The graph can be directed or not and the edges can have extra data. e.g. a name, a color, a weight, possibly even composite data in the form of tuples, though I haven't checked if that plays well with json. The vertices have to be hashable, e.g. strings. 
    """

    def __init__(self, graph_dict=None, file_path=None, red=None, black=None, directed=False):
        """
        Should be constructed from either a dict (of dicts of dicts) or from a json file serializing such a dict. The encoding is such that graph_dict[vertex1_color][vertex1][vertex2] is the edge data between vertex1 (of a certain color vertex1_color) and vertex2 (of the opposite color). A vertex v of color c without an edge is encoded by having graph_dict[c][v] be the empty dict. A directed graph is encoded as an undirected graph with edges in both directions.

        Args:
            - graph_dict: dict (of dicts of dicts) defining a bipartite graph. See example in module header. 
            - file_path: (str or pathlib.Path), path to .json containing such a dict. 
            - red: vertex color, defaults to 'red'.
            - black: vertex color, defaults to 'black.
            - directed: bool. If True the graph is directed, defaults to False.
        """        
        if file_path is not None:
            self.file_path=file_path
            self.directed=directed
            try:
                with open(file_path, "r") as read_file:
                    self._graph_dict = json.load(read_file)
            except FileNotFoundError as e:
                log.exception(f"Graph dict file {file_path} not found")
            except JSONDecodeError as e:
                log.exception(f"Json file {file_path} could not be decoded.")

            if len(self._graph_dict) != 2:
                raise Exception(f"__init__: input argument graph_dict must have outer dict be of length two!")
            # vertex colors
            self.red, self.black = tuple(self._graph_dict.keys())
            
        elif graph_dict is not None: # we are building the graph from a graph dict, so this determines the color 
            # make sure outermost dict has right shape
            if len(graph_dict) != 2:
                raise Exception(f"__init__: input argument graph_dict must have outer dict be of length two!")
            self._graph_dict = graph_dict
            self.directed=directed
            # vertex colors
            self.red, self.black = tuple(graph_dict.keys())
        elif red is not None or black is not None:
            if red is not None and black is not None:
                self.red = red
                self.black = black
            else:
                raise Exception(f"Error: Need either both or none of red and black specified!")
        else:
            self.red = 'red'
            self.black = 'black'
            self._graph_dict = empty_graph_dict(red=self.red, black=self.black)
            self.directed=directed
            self.file_path=None


    def colors(self):
        """Return list of vertex colors of the bipartite graph. """
        return [self.red, self.black]


    def __opposite_color__(self, color):
        """Return opposite vertex color. """

        if color == self.red:
            answer = self.black
        elif color == self.black:
            answer = self.red
        else:
            raise Exception(f"not a valid color: {color}")
        return answer


    def vertices(self, color):
        """Return the vertices of the graph. """
        return self._graph_dict[color].keys()


    def vertices_opposite(self, vertex, vertex_color, edge_data=None):
        """ 
        Return list of neighbouring vertices, i.e. the other 'heads' of the edges starting at vertex. Since our graph is bipartite, these are always of the opposite color, hence 'opposite'.
        """
        
        answer = self._graph_dict[vertex_color][vertex].keys()
        if edge_data is not None:
            answer = map(lambda opp_vertex_edge_data_pair: opp_vertex_edge_data_pair[0], 
                            filter(lambda opp_vertex_edge_data_pair: opp_vertex_edge_data_pair[1] == edge_data, 
                                    self._graph_dict[vertex_color][vertex].items()))
        return list(answer)


    def exists_vertex(self, vertex_name, vertex_color=None):
        """Check whether a vertex of a certain color exists. """

        # whether vertex exists in either color
        exists_color_red = vertex_name in self._graph_dict[self.red]
        exists_color_black = vertex_name in self._graph_dict[self.black]

        if vertex_color == None:
            answer = exists_color_red or exists_color_black
        elif vertex_color == self.red:
            answer = exists_color_red
        elif vertex_color == self.black:
            answer = exists_color_black
        else:
            log.error(f"Not a valid vertex_color: {vertex_color}. Graph vertex colors are {self.colors()}.")

        return answer


    def exists_edge(self, from_vertex, from_vertex_color, to_vertex, edge_data=None):
        """Check whether an edge exists. """

        if edge_data == None:
            answer = (to_vertex in self._graph_dict[from_vertex_color][from_vertex])
        else:
            answer = to_vertex in self.vertices_opposite(from_vertex, from_vertex_color, edge_data=edge_data)
        return answer


    def edge_data(self, from_vertex, from_color, to_vertex):
        """
        Return edge data.
        
        Args:
            - from_vertex:
            - from_color:
            - to_vertex: 
        """
        
        return self._graph_dict[from_color][from_vertex][to_vertex]


    def add_vertex(self, vertex_name, vertex_color):
        """
        Add a vertex. 
        
        Args:
            - vertex_name
            - vertex_color
        """

        # check if vertex already exists
        if self.exists_vertex(vertex_name):
            log.info(f"Vertex {vertex_name} of already exists!")
        else:
            # create vertex w/o edges
            self._graph_dict[vertex_color][vertex_name] = {}


    def add_edge(self, from_vertex, from_vertex_color, to_vertex, edge_data, force=False):
        """
        Add an edge. If the vertices do not yet exist will create them. Throws an error if an edge between the vertices already exists unless force is True, in which case it overwrites the existing edge_data.

        Args:
            - from_vertex:
            - from_color:
            - to_vertex: 
            - edge_data:
            - force: 
        """
        
        if not self.exists_vertex(from_vertex, from_vertex_color):
            log.info(f"add_edge: vertex {from_vertex} does not exist. Creating first...")
            self.add_vertex(from_vertex, from_vertex_color)
            #self.add_edge(from_vertex, from_vertex_color, to_vertex, edge_data, force)
        if not self.exists_vertex(to_vertex, self.__opposite_color__(from_vertex_color)):
            log.info(f"add_edge: vertex {to_vertex} does not exist. Creating first...")
            self.add_vertex(to_vertex, self.__opposite_color__(from_vertex_color))
            #self.add_edge(from_vertex, from_vertex_color, to_vertex, edge_data, force)
        if force == False and self.exists_edge(from_vertex, from_vertex_color, to_vertex):
            log.error(f"add_edge: an edge {from_vertex} (color: {from_vertex_color}) to {to_vertex} already exists. Set force=True to overwrite.")
            raise Exception(f"add_edge: an edge {from_vertex} (color: {from_vertex_color}) to {to_vertex} already exists. Set force=True to overwrite.")
        else:
            #add edge or update edge_date
            self._graph_dict[from_vertex_color][from_vertex][to_vertex] = edge_data
            
            #add or update opposite edge if graph is undirected:
            if self.directed == False: 
                to_vertex_color = self.__opposite_color__(from_vertex_color)
                self._graph_dict[to_vertex_color][to_vertex][from_vertex] = edge_data


    def delete_vertex(self, vertex_name, vertex_color, force_delete_with_edges=True):
        """
        Delete a vertex. If force==False, will delete only if the vertex has no edges. If force==True, will also delete edges starting or ending at the vertex. Note that in that case we don't delete dangling opposite vertices (i.e. opposite vertices that don't have any edges left after this). Only implemented for directed graphs.

        Args:
            - vertex_name:
            - vertex_color:
            - force_delete_with_edges:
        """

        if not self.exists_vertex(vertex_name, vertex_color):

            log.info(f"delete_vertex: nothing to do, vertex {vertex_name} does not exist.")

        # if force_delete_with_edges=False check if vertex has outgoing adjacent edges
        elif self.directed==True:
            
            log.error(f"Sorry, delete_vertex is not implemented for directed graphs. I was too lazy to code up the complication of checking which edges end in {vertex_name} :(")

            raise Exception(f"Sorry, delete_vertex is not implemented for directed graphs. I was too lazy to code up the complication of checking which edges end in  a given vertex :(")
        
        elif force_delete_with_edges==False and list(self.vertices_opposite(vertex_name, vertex_color)) != []:

            raise Exception(f"delete_vertex: vertex {vertex_name} of color {vertex_color} has edges. Set force_delete_with_edges=True to delete anyway (along with adjacent edges).")

        else:

            # thinking of an undirected graph as a directed graph where for each edge there is an opposite edge, 
            # we first take out the edges _ending_ in vertex, i.e. the opposite edges to the outgoing ones at vertex.
            opposite_color = self.__opposite_color__(vertex_color)
            for opposite_vertex in self._graph_dict[vertex_color][vertex_name].keys():
                self._graph_dict[opposite_color][opposite_vertex].pop(vertex_name)
            #then we take out the edges starting in vertex and the vertex itself
            self._graph_dict[vertex_color].pop(vertex_name)


    def delete_edge(self, from_vertex, from_vertex_color, to_vertex):
        """
        Delete the edge between vertex from_vertex of color from_vertex_color to to_vertex. 

        Args:
            - from_vertex (vertex):
            - from_vertex_color (color):
            - to_vertex (vertex):

        Returns:
            None

        """

        if not to_vertex in self._graph_dict[from_vertex_color][from_vertex]:
            log.info(f"delete_edge({from_vertex}, {from_vertex_color}, {to_vertex}): There is no such edge.")
        else:
            self._graph_dict[from_vertex_color][from_vertex].pop(to_vertex)
            if self.directed==False: #delete opposite edge
                opposite_color = self.__opposite_color__(from_vertex_color)
                self._graph_dict[opposite_color][to_vertex].pop(from_vertex)
    

    def save_to_file(self, file_path=None):
        """
        Save graph (i.e. graph_dict) to disk as json file.

        Args:
            - file_path (str or pathlib.Path): path of json file to save graph to.

        Returns:
            None
        """

        if file_path is None:
            if self.file_path is not None:
                file_path=self.file_path
            else:
                raise Exception(f"save_to_file: no file_path on record, specify as file_path argument.")
        else:
            self.file_path=file_path
            with open(file_path, "w") as write_file:
                json.dump(self._graph_dict, write_file)


    def really_undirected(self):
        """
        Check if graph is really undirected, i.e. if for each edge the opposite edge exists as well.
        Useful for testing.

        Returns:
            bool, True if graph is undirected, False if it's not.
        """
        # an empty graph is undirected
        answer = True
        
        try:
            for color in self._graph_dict.keys():
                opposite_color = self.__opposite_color__(color)
                for vertex in self._graph_dict[color]:
                    for opposite_vertex in self._graph_dict[color][vertex]:
                        if (self._graph_dict[color][vertex][opposite_vertex] != self._graph_dict[opposite_color][opposite_vertex][vertex]):
                            answer = False
        except KeyError as e:
            answer = False

        return answer


    def __eq__(self, other):
        """
        Check equality of graphs. Two graphs are equal if the vertex sets and colors and the edge sets and edge data agree, which is tested by asking whether the underlying dicts are equal.

        Args:
            - other (BipartiteGraph): 
        
        Returns:
            - (bool): True if the graphs are equal, False otherwise.
        """

        return self._graph_dict == other.__graph_dict__


    def __str__(self):
        """
        Return string representation of the graph.

        Args:
        Returns:
            - (str): string representation of the graph.
        """

        return json.dumps(self._graph_dict, indent=4)