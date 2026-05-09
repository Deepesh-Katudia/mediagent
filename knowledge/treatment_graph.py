"""
treatment_graph.py
==================
Weighted directed graph of treatment states used by TreatmentPlannerAgent
to find the lowest-cost care pathway via A* search (Russell & Norvig, Ch. 3-4).

This module defines:
- TREATMENT_STATES: nodes in the treatment graph, representing escalating
  levels of clinical intervention from initial diagnosis to full recovery.
- TREATMENT_HEURISTIC: admissible heuristic values (estimated cost-to-goal)
  for each state, used by the A* algorithm to guide the search efficiently.
- DISEASE_SEVERITY: per-disease severity classification (mild/moderate/severe)
  that determines which treatment state A* starts its search from.
- SEVERITY_START: maps a severity level to its recommended entry state in the
  treatment graph, so mild conditions skip directly to rest and severe ones
  begin at prescription medication.
- build_treatment_graph(): constructs and returns the NetworkX DiGraph whose
  edge weights represent the relative cost (invasiveness/risk) of each
  treatment transition.

AI technique supported: A* Search (TreatmentPlannerAgent).
"""

import networkx as nx

TREATMENT_STATES = [
    "diagnosed", "rest_prescribed", "otc_medication",
    "prescription_medication", "specialist_referral",
    "hospitalization", "treated"
]

TREATMENT_HEURISTIC = {
    "diagnosed": 5,
    "rest_prescribed": 3,
    "otc_medication": 2,
    "prescription_medication": 1,
    "specialist_referral": 1,
    "hospitalization": 1,
    "treated": 0
}

DISEASE_SEVERITY = {
    "Flu": "mild",
    "Common Cold": "mild",
    "Tension Headache": "mild",
    "Bronchitis": "moderate",
    "Hypertension": "moderate",
    "Migraine": "moderate",
    "Type 2 Diabetes": "moderate",
    "Hypothyroidism": "moderate",
    "Pneumonia": "severe",
    "Angina": "severe"
}

SEVERITY_START = {
    "mild": "rest_prescribed",
    "moderate": "otc_medication",
    "severe": "prescription_medication"
}


def build_treatment_graph() -> nx.DiGraph:
    """Build and return the treatment pathway graph.

    Constructs a weighted directed graph (DiGraph) where each node is a
    treatment state and each directed edge represents a possible clinical
    transition. Edge weights encode the cost (risk/invasiveness) of taking
    that step, so A* can prefer lower-cost pathways when multiple routes to
    the 'treated' state exist.

    Returns:
        nx.DiGraph: A directed graph with TREATMENT_STATES as nodes and
        weighted edges representing clinical transitions.
    """
    G = nx.DiGraph()
    for state in TREATMENT_STATES:
        G.add_node(state)
    edges = [
        ("diagnosed", "rest_prescribed", 1),
        ("rest_prescribed", "treated", 1),
        ("rest_prescribed", "otc_medication", 1),
        ("otc_medication", "treated", 2),
        ("otc_medication", "prescription_medication", 2),
        ("prescription_medication", "treated", 3),
        ("prescription_medication", "specialist_referral", 3),
        ("specialist_referral", "treated", 4),
        ("specialist_referral", "hospitalization", 5),
        ("hospitalization", "treated", 8),
    ]
    for src, dst, cost in edges:
        G.add_edge(src, dst, cost=cost)
    return G
