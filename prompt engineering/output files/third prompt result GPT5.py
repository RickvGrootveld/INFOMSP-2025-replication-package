import numpy as np
import networkx as nx

from qiskit.transpiler.exceptions import CouplingError


class CouplingMap:
    """
    Represents a directed coupling map of physical qubits on a quantum device.

    The coupling map defines how qubits are connected to each other. It is used
    during quantum circuit transpilation to determine the feasible routing of
    two-qubit gates based on hardware connectivity.
    """

    def __init__(self, couplinglist=None):
        """
        Initialize a CouplingMap object.

        Args:
            couplinglist (list[tuple[int, int]] or None): 
                A list of directed edges (source, target) representing couplings 
                between qubits. If None, an empty graph is created.
        """
        # Initialize a directed graph to represent qubit couplings
        self.graph = nx.DiGraph()
        # Cache for the qubit distance matrix
        self._dist_matrix = None
        # Cache for the list of qubits
        self._qubit_list = None

        # If an initial coupling list is provided, add the edges to the graph
        if couplinglist is not None:
            for source, target in couplinglist:
                self.add_edge(source, target)

    def size(self):
        """
        Return the number of physical qubits in the coupling map.

        Returns:
            int: Total number of nodes (qubits) in the graph.
        """
        return len(self.graph.nodes)

    def get_edges(self):
        """
        Retrieve all the coupling edges in the map.

        Returns:
            list[tuple[int, int]]: List of directed edges (source, target).
        """
        return [edge for edge in self.graph.edges()]

    def add_physical_qubit(self, physical_qubit):
        """
        Add a new physical qubit to the coupling map.

        Args:
            physical_qubit (int): The index of the physical qubit to add.

        Raises:
            CouplingError: If the input is not an integer or the qubit already exists.
        """
        # Ensure the qubit identifier is an integer
        if not isinstance(physical_qubit, int):
            raise CouplingError("Physical qubits should be integers.")
        # Prevent adding duplicate qubits
        if physical_qubit in self.physical_qubits:
            raise CouplingError(
                "The physical qubit %s is already in the coupling graph" % physical_qubit)
        # Add the qubit as a new node to the graph
        self.graph.add_node(physical_qubit)
        # Reset caches since the graph structure has changed
        self._dist_matrix = None  
        self._qubit_list = None 

    def add_edge(self, src, dst):
        """
        Add a directed coupling (edge) between two physical qubits.

        Args:
            src (int): Source qubit index.
            dst (int): Destination qubit index.

        Notes:
            If a qubit does not exist in the map, it will be added automatically.
        """
        # Ensure both qubits are present in the graph
        if src not in self.physical_qubits:
            self.add_physical_qubit(src)
        if dst not in self.physical_qubits:
            self.add_physical_qubit(dst)
        # Add the directed edge representing a coupling
        self.graph.add_edge(src, dst)
        # Reset distance matrix cache as the topology has changed
        self._dist_matrix = None 

    def subgraph(self, nodelist):
        """
        Create a new coupling map containing only a subset of qubits.

        Args:
            nodelist (list[int]): List of qubits to include in the subgraph.

        Returns:
            CouplingMap: A smaller coupling map containing only the specified nodes.
        """
        # Create a new CouplingMap object
        subcoupling = CouplingMap()
        # Extract the subgraph for the given nodes
        subcoupling.graph = self.graph.subgraph(nodelist)
        # Ensure all nodes are added to the subcoupling
        for node in nodelist:
            if node not in subcoupling.physical_qubits:
                subcoupling.add_physical_qubit(node)
        return subcoupling

    @property
    def physical_qubits(self):
        """
        Return the list of all physical qubits in the coupling map.

        Returns:
            list[int]: Sorted list of all qubit indices.
        """
        # Cache the list of qubits to avoid recomputation
        if self._qubit_list is None:
            self._qubit_list = sorted([pqubit for pqubit in self.graph.nodes])
        return self._qubit_list

    def is_connected(self):
        """
        Check if the coupling map is weakly connected.

        Returns:
            bool: True if the graph is weakly connected, False otherwise.
        """
        # Use NetworkX to verify connectivity
        try:
            return nx.is_weakly_connected(self.graph)
        except nx.exception.NetworkXException:
            return False

    def _compute_distance_matrix(self):
        """
        Compute and store the shortest-path distance matrix for all pairs of qubits.

        Raises:
            CouplingError: If the coupling graph is not connected.
        """
        # Ensure that the coupling map is connected before computing distances
        if not self.is_connected():
            raise CouplingError("coupling graph not connected")
        # Compute all-pairs shortest path lengths using the undirected graph
        lengths = nx.all_pairs_shortest_path_length(self.graph.to_undirected(as_view=True))
        lengths = dict(lengths)
        size = len(lengths)
        # Initialize the matrix with zeros
        cmap = np.zeros((size, size))
        # Fill the matrix with path lengths
        for idx in range(size):
            cmap[idx, np.fromiter(lengths[idx].keys(), dtype=int)] = np.fromiter(
                lengths[idx].values(), dtype=int)
        # Store the computed matrix
        self._dist_matrix = cmap

    def distance(self, physical_qubit1, physical_qubit2):
        """
        Get the shortest distance between two qubits.

        Args:
            physical_qubit1 (int): The first qubit.
            physical_qubit2 (int): The second qubit.

        Returns:
            int: The shortest path distance between the two qubits.

        Raises:
            CouplingError: If either qubit is not part of the coupling map.
        """
        # Ensure both qubits exist in the coupling graph
        if physical_qubit1 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit1,))
        if physical_qubit2 not in self.physical_qubits:
            raise CouplingError("%s not in coupling graph" % (physical_qubit2,))
        # Compute the distance matrix if it hasn't been computed yet
        if self._dist_matrix is None:
            self._compute_distance_matrix()
        # Return the shortest distance between the two qubits
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
            CouplingError: If there is no path connecting the two qubits.
        """
        # Attempt to find the shortest path between the qubits
        try:
            return nx.shortest_path(self.graph.to_undirected(as_view=True), source=physical_qubit1,
                                    target=physical_qubit2)
        # Raise an error if the qubits are not connected
        except nx.exception.NetworkXNoPath:
            raise CouplingError(
                "Nodes %s and %s are not connected" % (str(physical_qubit1), str(physical_qubit2)))

    def __str__(self):
        """
        Return a string representation of the coupling map.

        Returns:
            str: A formatted string listing all directed edges.
        """
        # Build a string showing all edges in the coupling map
        string = ""
        if self.get_edges():
            string += "["
            string += ", ".join(["[%s, %s]" % (src, dst) for (src, dst) in self.get_edges()])
            string += "]"
        return string
