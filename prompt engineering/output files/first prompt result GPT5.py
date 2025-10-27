import numpy as np
import networkx as nx

from qiskit.transpiler.exceptions import CouplingError


class CouplingMap:
    """
    Represents a directed coupling map for quantum devices.
    The coupling map defines the connectivity (coupling relationships)
    between physical qubits, typically used in quantum circuit transpilation.
    """

    def __init__(self, couplinglist=None):
        """
        Initialize a CouplingMap object.

        Args:
            couplinglist (list[tuple[int, int]] or None): 
                Optional list of edges representing connections between physical qubits.
                Each edge is a tuple (source, target).
        """
        self.graph = nx.DiGraph()
        self._dist_matrix = None
        self._qubit_list = None

        if couplinglist is not None:
            for source, target in couplinglist:
                self.add_edge(source, target)

    def size(self):
        """
        Return the number of physical qubits in the coupling map.

        Returns:
            int: Number of qubits (nodes) in the coupling graph.
        """
        return len(self.graph.nodes)

    def get_edges(self):
        """
        Retrieve all edges in the coupling map.

        Returns:
            list[tuple[int, int]]: A list of directed edges between physical qubits.
        """
        return [edge for edge in self.graph.edges()]

    def add_physical_qubit(self, physical_qubit):
        """
        Add a physical qubit (node) to the coupling map.

        Args:
            physical_qubit (int): Identifier of the physical qubit to add.

        Raises:
            CouplingError: If the input is not an integer or the qubit already exists.
        """
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        if physical_qubit in self.physical_qubits:
            raise CouplingError(
                "The physical qubit %s is already in the coupling graph" % physical_qubit)
        self.graph.add_node(physical_qubit)
        self._dist_matrix = None  
        self._qubit_list = None 

    def add_edge(self, src, dst):
        """
        Add a directed edge (coupling) between two physical qubits.

        Args:
            src (int): Source qubit.
            dst (int): Destination qubit.

        Notes:
            If either qubit is not already in the map, it will be added automatically.
        """
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        self.graph.add_edge(src, dst)
        self._dist_matrix = None 

    def subgraph(self, nodelist):
        """
        Create a subgraph (smaller coupling map) containing only the specified qubits.

        Args:
            nodelist (list[int]): List of qubits to include in the subgraph.

        Returns:
            CouplingMap: A new coupling map representing the subgraph.
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
        Return the list of all physical qubits in the coupling map.

        Returns:
            list[int]: Sorted list of all physical qubits (nodes).
        """
        if self._qubit_list is None:
            self._qubit_list = sorted([pqubit for pqubit in self.graph.nodes])
        return self._qubit_list

    def is_connected(self):
        """
        Check whether the coupling map is weakly connected.

        Returns:
            bool: True if the graph is weakly connected, False otherwise.
        """
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False

    def _compute_distance_matrix(self):
        """
        Compute and store the shortest-path distance matrix between all physical qubits.

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
        Return the shortest distance between two physical qubits.

        Args:
            physical_qubit1 (int): The first qubit.
            physical_qubit2 (int): The second qubit.

        Returns:
            int: The number of edges in the shortest path between the two qubits.

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
        Compute the shortest undirected path between two physical qubits.

        Args:
            physical_qubit1 (int): The first qubit.
            physical_qubit2 (int): The second qubit.

        Returns:
            list[int]: Ordered list of qubits representing the shortest path.

        Raises:
            CouplingError: If there is no path between the two qubits.
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
            str: A formatted string listing all directed edges.
        """
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join(["[%s, %s]" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string
