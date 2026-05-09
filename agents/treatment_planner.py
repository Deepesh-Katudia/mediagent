"""
TreatmentPlannerAgent — A* Search on a Treatment Graph
=======================================================

AI Technique: A* heuristic search (via NetworkX ``astar_path``)

Role in pipeline:
    The TreatmentPlannerAgent is invoked by the OrchestratorAgent after a
    confident diagnosis has been made.  Given the top-ranked disease, it
    determines the most cost-effective care pathway through a weighted
    directed graph of treatment options, producing an ordered sequence of
    steps from an initial intervention to the ``"treated"`` terminal node.

    Graph structure (defined in ``knowledge.treatment_graph``):
        Nodes represent treatment modalities such as ``"rest_prescribed"``,
        ``"otc_medication"``, ``"prescription_medication"``,
        ``"specialist_referral"``, and ``"hospitalization"``.  Directed edges
        carry a ``cost`` attribute encoding the clinical effort or resource
        burden of transitioning between steps.  All paths ultimately terminate
        at the ``"treated"`` sink node.

    Search strategy:
        A* is preferred over simpler BFS/Dijkstra because it allows the
        ``TREATMENT_HEURISTIC`` table to encode domain knowledge about how many
        additional steps each node is likely to require, guiding the search
        toward lower-cost pathways without exhaustive exploration.

    Severity-aware start node:
        The disease's severity level (``DISEASE_SEVERITY``) determines the
        entry point into the graph via ``SEVERITY_START``.  Mild diseases begin
        at rest/OTC nodes; severe diseases enter at prescription or
        hospitalisation nodes, reflecting clinical triage logic.
"""

import networkx as nx
from knowledge.treatment_graph import (
    build_treatment_graph, DISEASE_SEVERITY, SEVERITY_START,
    TREATMENT_HEURISTIC
)


class TreatmentPlannerAgent:
    """A* search agent that plans an ordered treatment pathway for a given disease.

    The agent holds a single pre-built NetworkX ``DiGraph`` representing the
    treatment option graph.  All planning calls operate on this shared graph,
    making the agent stateless with respect to individual sessions.

    Class attributes
    ----------------
    DEFAULT_START : str
        Fallback entry node used when a disease's severity level is not found
        in ``SEVERITY_START``.  Defaults to ``"otc_medication"``, representing
        a conservative, low-acuity starting point.
    """

    DEFAULT_START = "otc_medication"

    def __init__(self):
        """Build the treatment graph from the knowledge base at instantiation time."""
        self._graph = build_treatment_graph()

    def _heuristic(self, node: str, target: str) -> int:
        """Return the A* heuristic estimate of remaining cost from ``node`` to ``target``.

        Looks up the pre-computed heuristic value for ``node`` in the
        ``TREATMENT_HEURISTIC`` table.  If the node is not listed, returns 3
        as a conservative over-estimate (A* remains admissible as long as this
        value does not exceed the true remaining cost, which the table is
        designed to guarantee).

        Parameters
        ----------
        node : str
            The current graph node being evaluated.
        target : str
            The goal node (always ``"treated"`` in this pipeline).

        Returns
        -------
        int
            Heuristic estimate of the number of additional steps to the goal.
        """
        return TREATMENT_HEURISTIC.get(node, 3)

    def plan_treatment(self, disease: str) -> list:
        """Compute the optimal treatment pathway for the given disease using A*.

        Selects the graph entry point based on the disease's severity level,
        then runs ``nx.astar_path`` to find the minimum-cost sequence of
        treatment nodes leading to ``"treated"``.

        If no path exists in the graph (e.g. a disconnected subgraph due to a
        data inconsistency), the method falls back to a safe default pathway:
        start node → specialist referral → treated.

        Parameters
        ----------
        disease : str
            The canonical disease identifier (e.g. ``"influenza"``,
            ``"hypertension"``).

        Returns
        -------
        list[str]
            Ordered list of treatment step node names, starting from the
            severity-appropriate entry node and ending with ``"treated"``.
        """
        severity = DISEASE_SEVERITY.get(disease, "moderate")
        start = SEVERITY_START.get(severity, self.DEFAULT_START)

        try:
            path = nx.astar_path(
                self._graph,
                source=start,
                target="treated",
                heuristic=self._heuristic,
                weight="cost"
            )
            return path
        except nx.NetworkXNoPath:
            return [start, "specialist_referral", "treated"]
