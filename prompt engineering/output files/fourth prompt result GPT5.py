import numpy as np
import networkx as nx

from qiskit.transpiler.exceptions import CouplingError


class CouplingMap:
    """
    Represents a directed coupling map of physical qubits in a quantum device.

    This class defines the connectivity between physical qubits, typically used in quantum circuit 
    transpilation to determine which qubits can interact directly. It allows adding qubits and edges, 
    checking connectivity, and computing distances between qubits.
    """

    def __init__(self, couplinglist=None):
        """
        Initialize a CouplingMap instance.

        Args:
            couplinglist (list[tuple[int, int]] or None): 
                Optional list of tuples defining directed connections (edges) 
                between physical qubits. Each tuple (source, target) represents 
                a coupling between two qubits.
        """
        # Create a directed graph to store the couplings
        self.graph = nx.DiGraph()
        # Initialize distance matrix cache
        self._dist_matrix = None
        # Initialize physical qubit list cache
        self._qubit_list = None

        # If a list of couplings is provided, add them to the graph
        if couplinglist is not None:
            for source, target in couplinglist:
                self.add_edge(source, target)

    def size(self):
        """
        Get the total number of physical qubits in the coupling map.

        Returns:
            int: The number of qubits (nodes) in the coupling graph.
        """
        return len(self.graph.nodes)

    def get_edges(self):
        """
        Retrieve all directed edges in the coupling map.

        Returns:
            list[tuple[int, int]]: A list of directed connections (edges) between qubits.
        """
        return [edge for edge in self.graph.edges()]

    def add_physical_qubit(self, physical_qubit):
        """
        Add a physical qubit (node) to the coupling map.

        Args:
            physical_qubit (int): The integer identifier of the qubit to add.

        Raises:
            CouplingError: If the qubit is not an integer or already exists in the graph.
        """
        # Ensure that the input is a valid integer
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        # Prevent adding the same qubit twice
        if physical_qubit in self.physical_qubits:
            raise CouplingError(
                "The physical qubit %s is already in the coupling graph" % physical_qubit)
        # Add the qubit as a node in the graph
        self.graph.add_node(physical_qubit)
        # Invalidate cached data as the structure has changed
        self._dist_matrix = None  
        self._qubit_list = None 

    def add_edge(self, src, dst):
        """
        Add a directed edge (coupling) between two physical qubits.

        Args:
            src (int): The source qubit.
            dst (int): The destination qubit.

        Notes:
            If a qubit is not yet part of the coupling map, it is automatically added.
        """
        # Add source qubit if not already in the graph
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        # Add destination qubit if not already in the graph
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        # Add the directed edge between qubits
        self.graph.add_edge(src, dst)
        # Reset cached distance matrix
        self._dist_matrix = None 

    def subgraph(self, nodelist):
        """
        Create a new CouplingMap object as a subgraph of the current map.

        Args:
            nodelist (list[int]): A list of qubits to include in the subgraph.

        Returns:
            CouplingMap: A new CouplingMap containing only the specified qubits.
        """
        # Create a new CouplingMap instance
        subcoupling = CouplingMap()
        # Generate a subgraph from the selected nodes
        subcoupling.graph = self.graph.subgraph(nodelist)
        # Ensure all nodes exist in the new subcoupling
        for node in nodelist:
            if node not in subcoupling.physical_qubits:
                subcoupling.add_physical_qubit(node)
        return subcoupling

    @property
    def physical_qubits(self):
        """
        Retrieve a sorted list of physical qubits in the coupling map.

        Returns:
            list[int]: A sorted list of all physical qubit identifiers.
        """
        # Cache the list for efficiency
        if self._qubit_list is None:
            self._qubit_list = sorted([pqubit for pqubit in self.graph.nodes])
        return self._qubit_list

    def is_connected(self):
        """
        Check if the coupling map is weakly connected.

        Returns:
            bool: True if the graph is weakly connected (every node is reachable 
            via undirected paths), False otherwise.
        """
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False

    def _compute_distance_matrix(self):
        """
        Compute and store the shortest-path distance matrix between all qubits.

        Raises:
            CouplingError: If the coupling graph is not connected.
        """
        # Ensure the coupling graph is connected before computing distances
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        # Compute the shortest path lengths for all qubit pairs
        lengths = nx.all_pairs_shortest_path_length(self.graph.to_undirected(as_view=True))
        lengths = dict(lengths)
        size = len(lengths)
        # Initialize a distance matrix of zeros
        cmap = np.zeros((size, size))
        # Populate the matrix with computed shortest path lengths
        for idx in range(size):
            cmap[idx, np.fromiter(lengths[idx].keys(), dtype=int)] = np.fromiter(
                lengths[idx].values(), dtype=int)
        # Cache the computed distance matrix
        self._dist_matrix = cmap

    def distance(self, physical_qubit1, physical_qubit2):
        """
        Compute the shortest distance between two qubits in the coupling map.

        Args:
            physical_qubit1 (int): The first qubit.
            physical_qubit2 (int): The second qubit.

        Returns:
            int: The number of edges in the shortest path between the two qubits.

        Raises:
            CouplingError: If either qubit is not present in the coupling graph.
        """
        # Check that both qubits exist in the graph
        if physical_qubit1 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit1,))
        if physical_qubit2 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit2,))
        # Compute the distance matrix if it hasn't been created yet
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        # Return the computed distance between the two qubits
        return self._dist_matrix[physical_qubit1, physical_qubit2]

    def shortest_undirected_path(self, physical_qubit1, physical_qubit2):
        """
        Find the shortest undirected path between two qubits.

        Args:
            physical_qubit1 (int): The first qubit.
            physical_qubit2 (int): The second qubit.

        Returns:
            list[int]: A list of qubits representing the shortest path.

        Raises:
            CouplingError: If the qubits are not connected.
        """
        try:
            # Use NetworkX to find the shortest undirected path
            return nx.shortest_path(self.graph.to_undirected(as_view=True), source=physical_qubit1,
                                    target=physical_qubit2)
        except nx.exception.NetworkXNoPath:
            # Raise an error if no path exists
            raise CouplingError(
                "Nodes %s and %s are not connected" % (str(physical_qubit1), str(physical_qubit2)))

    def __str__(self):
        """
        Return a string representation of the coupling map.

        Returns:
            str: A formatted string showing all directed edges between qubits.
        """
        string = ""
        # Build the string representation of all edges
        if self.get_edges():
            string += "["
            string += ", ".join(["[%s, %s]" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string
