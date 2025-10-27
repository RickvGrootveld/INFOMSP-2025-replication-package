import numpy as np
import networkx as nx

from qiskit.transpiler.exceptions import CouplingError


class CouplingMap:
    """
    Represents the coupling map of a quantum device, defining which qubits are connected.
    
    This class uses a directed graph to model the connectivity between physical qubits,
    allowing for distance calculations and path finding between qubits.
    """

    def __init__(self, couplinglist=None):
        """
        Initialize a CouplingMap instance.
        
        Args:
            couplinglist: Optional list of [source, target] pairs representing edges
                         between physical qubits. If None, creates an empty coupling map.
        """
        self.graph = nx.DiGraph()
        self._dist_matrix = None
        self._qubit_list = None

        if couplinglist is not None:
            for source, target in couplinglist:
                self.add_edge(source, target)

    def size(self):
        """
        Get the number of physical qubits in the coupling map.
        
        Returns:
            int: The number of nodes (qubits) in the coupling graph.
        """
        return len(self.graph.nodes)

    def get_edges(self):
        """
        Get all edges in the coupling map.
        
        Returns:
            list: A list of tuples representing edges between physical qubits.
        """
        return [edge for edge in self.graph.edges()]

    def add_physical_qubit(self, physical_qubit):
        """
        Add a physical qubit to the coupling map.
        
        Args:
            physical_qubit: Integer identifier for the physical qubit to add.
            
        Raises:
            CouplingError: If physical_qubit is not an integer or already exists in the graph.
        """
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        if physical_qubit in self.physical_qubits:
            raise CouplingError(
                "The physical qubit %s is already in the coupling graph" % physical_qubit)
        self.graph.add_node(physical_qubit)
        # Invalidate cached values since graph structure changed
        self._dist_matrix = None  
        self._qubit_list = None 

    def add_edge(self, src, dst):
        """
        Add an edge between two physical qubits in the coupling map.
        
        If either qubit doesn't exist in the graph, it will be added automatically.
        
        Args:
            src: Integer identifier for the source qubit.
            dst: Integer identifier for the destination qubit.
        """
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        self.graph.add_edge(src, dst)
        # Invalidate cached distance matrix since graph structure changed
        self._dist_matrix = None 

    def subgraph(self, nodelist):
        """
        Create a subgraph containing only the specified nodes.
        
        Args:
            nodelist: List of node identifiers to include in the subgraph.
            
        Returns:
            CouplingMap: A new CouplingMap instance containing only the specified nodes.
        """
        subcoupling = CouplingMap()
        subcoupling.graph = self.graph.subgraph(nodelist)
        for node in nodelist:
            if node not in subcoupling.physical_qubits:
                subcoupling.add_physical_qubit(node)
        return subcoupling

    @property
    def physical_qubits(self):
        """
        Get a sorted list of all physical qubits in the coupling map.
        
        Returns:
            list: Sorted list of integer identifiers for all physical qubits.
        """
        if self._qubit_list is None:
            self._qubit_list = sorted([pqubit for pqubit in self.graph.nodes])
        return self._qubit_list

    def is_connected(self):
        """
        Check if the coupling graph is weakly connected.
        
        A graph is weakly connected if there is a path between every pair of nodes
        when edge directions are ignored.
        
        Returns:
            bool: True if the graph is weakly connected, False otherwise.
        """
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False

    def _compute_distance_matrix(self):
        """
        Compute the distance matrix for all pairs of qubits.
        
        Creates a matrix where each element [i,j] represents the shortest path length
        between qubit i and qubit j in the undirected version of the graph.
        
        Raises:
            CouplingError: If the coupling graph is not connected.
        """
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        lengths = nx.all_pairs_shortest_path_length(self.graph.to_undirected(as_view=True))
        lengths = dict(lengths)
        size = len(lengths)
        cmap = np.zeros((size, size))
        for idx in range(size):
            cmap[idx, np.fromiter(lengths[idx].keys(), dtype=int)] = np.fromiter(
                lengths[idx].values(), dtype=int)
        self._dist_matrix = cmap

    def distance(self, physical_qubit1, physical_qubit2):
        """
        Calculate the shortest distance between two physical qubits.
        
        Args:
            physical_qubit1: Integer identifier for the first qubit.
            physical_qubit2: Integer identifier for the second qubit.
            
        Returns:
            float: The shortest path length between the two qubits.
            
        Raises:
            CouplingError: If either qubit is not in the coupling graph.
        """
        if physical_qubit1 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit1,))
        if physical_qubit2 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit2,))
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        return self._dist_matrix[physical_qubit1, physical_qubit2]

    def shortest_undirected_path(self, physical_qubit1, physical_qubit2):
        """
        Find the shortest path between two qubits in the undirected graph.
        
        Args:
            physical_qubit1: Integer identifier for the source qubit.
            physical_qubit2: Integer identifier for the target qubit.
            
        Returns:
            list: A list of qubit identifiers representing the shortest path.
            
        Raises:
            CouplingError: If no path exists between the two qubits.
        """
        try:
            return nx.shortest_path(self.graph.to_undirected(as_view=True), source=physical_qubit1,
                                    target=physical_qubit2)
        except nx.exception.NetworkXNoPath:
            raise CouplingError(
                "Nodes %s and %s are not connected" % (str(physical_qubit1), str(physical_qubit2)))

    def __str__(self):
        """
        Return a string representation of the coupling map.
        
        Returns:
            str: A string showing all edges in the format "[[src1, dst1], [src2, dst2], ...]".
        """
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join(["[%s, %s]" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string