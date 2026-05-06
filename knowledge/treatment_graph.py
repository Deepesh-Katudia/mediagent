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
