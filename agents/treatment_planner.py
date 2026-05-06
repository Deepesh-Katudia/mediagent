import networkx as nx
from knowledge.treatment_graph import (
    build_treatment_graph, DISEASE_SEVERITY, SEVERITY_START,
    TREATMENT_HEURISTIC
)


class TreatmentPlannerAgent:
    DEFAULT_START = "otc_medication"

    def __init__(self):
        self._graph = build_treatment_graph()

    def _heuristic(self, node: str, target: str) -> int:
        return TREATMENT_HEURISTIC.get(node, 3)

    def plan_treatment(self, disease: str) -> list:
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
