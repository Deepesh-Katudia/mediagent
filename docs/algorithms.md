# MediAgent — Algorithm Reference

**Team Members**: Deepesh Katudia, Aditya Srivastava, Manas Mankar, Vishnu Sai Reddy Alla, Bella Gerken.

This document provides a detailed description of each AI technique used in the MediAgent system, including its theoretical basis, concrete implementation details, and the role it plays in the diagnostic pipeline.

All algorithm references use the notation of Russell & Norvig, *Artificial Intelligence: A Modern Approach* (4th ed.).

---

## 1. First-Order Logic (FOL) + Forward Chaining

**Agent:** `DiagnosisAgent` (`agents/diagnosis_agent.py`)
**Knowledge base:** `knowledge/disease_rules.py`

### 1.1 Knowledge Base Structure

The knowledge base represents medical diagnostic rules as a dictionary of disease objects. Each disease entry contains two symptom sets that together encode the logical preconditions for that diagnosis:

```python
DISEASE_RULES = {
    "Flu": {
        "required":   ["fever", "body_aches", "fatigue"],
        "supporting": ["chills", "headache", "cough", "sore_throat"]
    },
    "Pneumonia": {
        "required":   ["fever", "cough", "chest_pain"],
        "supporting": ["shortness_of_breath", "fatigue", "sweating", "chills"]
    },
    # ... 10 diseases total
}
```

- **Required symptoms** are necessary conditions. If none are present, the disease is immediately ineligible regardless of supporting symptom overlap.
- **Supporting symptoms** are probabilistically associated but not individually necessary. They contribute additional confidence when present.

The global symptom vocabulary (`SYMPTOMS`) contains 30 canonical tokens that serve as proposition atoms for FOL inference and as feature columns for the Naive Bayes classifier.

### 1.2 Inference Algorithm

The forward chaining inference procedure maps the patient's collected symptom set against every disease rule and produces a coverage score:

```
FUNCTION run_fol_inference(symptoms):
    symptom_set ← set(symptoms)
    results ← []

    FOR EACH (disease, rules) IN DISEASE_RULES:
        required    ← set(rules["required"])
        supporting  ← set(rules["supporting"])

        required_matches    ← |required ∩ symptom_set|
        supporting_matches  ← |supporting ∩ symptom_set|

        # Hard gate: at least one required symptom must be present
        IF required_matches == 0:
            CONTINUE

        # Weighted FOL score
        fol_score ← (required_matches / |required|) × 0.7
                   + (supporting_matches / max(|supporting|, 1)) × 0.3

        APPEND {disease, fol_score, matched_required, matched_supporting}
              TO results

    RETURN top-3 results sorted by fol_score descending
```

The procedure is a form of forward chaining: observed symptoms (ground facts) are matched against rule antecedents, and matching diseases are promoted as candidate conclusions. Diseases that do not fire at least one required antecedent are pruned before scoring.

### 1.3 Confidence Computation

The FOL score is a weighted linear combination of two coverage ratios:

```
fol_score = (required_matches / |required|) × 0.70
           + (supporting_matches / |supporting|) × 0.30
```

The **70 / 30 split** encodes a domain assumption: required symptoms are diagnostically more informative (higher positive predictive value) than supporting symptoms, which are associated but not pathognomonic. Both terms are normalised to [0, 1] so the composite score is always in [0, 1].

**Example — Flu with symptoms {fever, body_aches, fatigue, headache}:**

- required = {fever, body_aches, fatigue}, required_matches = 3 → 3/3 = 1.0
- supporting = {chills, headache, cough, sore_throat}, supporting_matches = 1 → 1/4 = 0.25
- fol_score = 1.0 × 0.7 + 0.25 × 0.3 = **0.775**

---

## 2. Naive Bayes Classifier

**Agent:** `DiagnosisAgent` (`agents/diagnosis_agent.py`)
**Training:** `ml/train.py`, `ml/dataset_generator.py`
**Artefact:** `ml/model.pkl`

### 2.1 Feature Vector Encoding

Each patient symptom set is encoded as a binary presence/absence vector aligned to the 30-element `SYMPTOMS` vocabulary:

```
SYMPTOMS = ["fever", "cough", "fatigue", "body_aches", ..., "memory_issues"]
           (30 canonical symptom tokens)

vector[i] = 1   if SYMPTOMS[i] ∈ patient_symptoms
           = 0   otherwise
```

This produces a sparse 30-dimensional binary vector that is passed directly to the `MultinomialNB.predict_proba` method. Multinomial Naive Bayes is appropriate here because each feature is a count variable over a discrete vocabulary (in the binary case, counts are 0 or 1).

### 2.2 Synthetic Training Data

Because real patient data is unavailable for this academic prototype, the training set is generated programmatically from the same FOL rule base used for symbolic inference:

```
FUNCTION generate_dataset(samples_per_disease=1000, noise_prob=0.1):
    FOR EACH disease IN DISEASE_RULES:
        FOR i IN 1..samples_per_disease:
            row ← {symptom: 0 FOR symptom IN SYMPTOMS}

            # Required symptoms are always present
            FOR s IN rules["required"]:
                row[s] ← 1

            # Supporting symptoms present with 60% probability
            FOR s IN rules["supporting"]:
                IF random() > 0.4:
                    row[s] ← 1

            # Add 10% random bit-flip noise to prevent memorisation
            FOR s IN SYMPTOMS:
                IF random() < noise_prob:
                    row[s] ← 1 - row[s]

            APPEND row TO dataset

    RETURN dataset
```

This design ensures the classifier learns the probabilistic structure of the rule base (required symptoms near-always present, supporting symptoms present ~60% of the time) while the noise injection prevents the model from achieving a trivially perfect fit that would fail to generalise.

### 2.3 Train/Test Split and Accuracy Threshold

The training script uses an **80/20 train/test split** (`test_size=0.2`). The random state is fixed at 42 for reproducibility.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
assert accuracy_score(y_test, model.predict(X_test)) >= 0.75
```

The 0.75 accuracy threshold is enforced as a hard assertion; the training script will raise `AssertionError` and refuse to write `model.pkl` if accuracy falls below this floor. The trained model is serialised with `joblib.dump` and loaded at agent initialisation time.

Note: The original code comment in `train.py` describes a 70/30 split, but the implementation uses `test_size=0.2` (80/20 split). The 80/20 split is what is actually enforced at runtime.

### 2.4 Prediction and Blending with FOL

The ML inference method encodes the symptom list and queries the classifier's posterior:

```python
def run_ml_inference(self, symptoms):
    vector = [[1 if s in symptoms else 0 for s in SYMPTOMS]]
    probs = model.predict_proba(vector)[0]
    return {cls: float(prob) for cls, prob in zip(model.classes_, probs)}
```

The FOL and ML scores are blended using a fixed weighted average:

```
confidence = FOL_WEIGHT × fol_score + ML_WEIGHT × ml_prob
           = 0.6 × fol_score + 0.4 × ml_prob
```

The **60% FOL / 40% ML** weighting deliberately privileges the interpretable symbolic layer. This ensures that:

1. The ML layer alone cannot surface a disease that failed the FOL hard gate (required symptom check). A disease with `fol_score = 0` (no required symptoms matched) is never added to the candidate list, so its `ml_prob` never contributes.
2. Clinical knowledge encoded by domain experts is weighted above the statistical signal, which is derived from synthetic data.

If no pre-trained model is available at `ml/model.pkl`, `ml_prob` defaults to 0.0 for all candidates and the system operates in pure FOL mode.

---

## 3. A* Search

**Agent:** `TreatmentPlannerAgent` (`agents/treatment_planner.py`)
**Knowledge base:** `knowledge/treatment_graph.py`

### 3.1 State Space — Treatment Graph

The treatment state space is a weighted directed graph (`NetworkX DiGraph`) whose nodes represent clinical intervention levels and whose directed edges represent feasible transitions between them. Edge weights encode the relative cost (invasiveness, resource burden) of each transition.

**Nodes:**

| Node | Clinical Meaning |
|---|---|
| `diagnosed` | Diagnosis confirmed, no treatment yet |
| `rest_prescribed` | Self-care: rest and hydration |
| `otc_medication` | Over-the-counter medication |
| `prescription_medication` | Prescription drug required |
| `specialist_referral` | Referral to a specialist |
| `hospitalization` | Inpatient care required |
| `treated` | Recovery expected (goal node) |

**Edges and costs:**

```
diagnosed              → rest_prescribed          cost = 1
rest_prescribed        → treated                  cost = 1
rest_prescribed        → otc_medication           cost = 1
otc_medication         → treated                  cost = 2
otc_medication         → prescription_medication  cost = 2
prescription_medication → treated                 cost = 3
prescription_medication → specialist_referral     cost = 3
specialist_referral    → treated                  cost = 4
specialist_referral    → hospitalization          cost = 5
hospitalization        → treated                  cost = 8
```

### 3.2 Heuristic Function

The A* heuristic is a pre-computed lookup table (`TREATMENT_HEURISTIC`) that estimates the minimum remaining cost from each node to the goal (`"treated"`):

```python
TREATMENT_HEURISTIC = {
    "diagnosed":               5,
    "rest_prescribed":         3,
    "otc_medication":          2,
    "prescription_medication": 1,
    "specialist_referral":     1,
    "hospitalization":         1,
    "treated":                 0,
}
```

This heuristic is **admissible** (never overestimates) because the values are lower bounds on the actual path cost from each node to the goal. Admissibility guarantees that A* finds the optimal (minimum-cost) path.

The heuristic function used by `nx.astar_path`:

```python
def _heuristic(self, node, target):
    return TREATMENT_HEURISTIC.get(node, 3)  # default 3 if unknown node
```

### 3.3 Cost Function

The path cost is the cumulative sum of edge `cost` attributes along the chosen path. A* selects the path that minimises `f(n) = g(n) + h(n)` where `g(n)` is the cumulative edge cost from the start node to node `n`, and `h(n)` is the heuristic estimate above.

### 3.4 Severity-Aware Start Node

The entry point into the treatment graph is determined by the disease's clinical severity, not fixed at `"diagnosed"`. This reflects triage logic: mild conditions begin at self-care, while severe conditions bypass conservative steps and enter at prescription or hospitalisation:

```python
DISEASE_SEVERITY = {
    "Flu":              "mild",
    "Common Cold":      "mild",
    "Tension Headache": "mild",
    "Bronchitis":       "moderate",
    "Hypertension":     "moderate",
    "Migraine":         "moderate",
    "Type 2 Diabetes":  "moderate",
    "Hypothyroidism":   "moderate",
    "Pneumonia":        "severe",
    "Angina":           "severe",
}

SEVERITY_START = {
    "mild":     "rest_prescribed",
    "moderate": "otc_medication",
    "severe":   "prescription_medication",
}
```

### 3.5 Example Path

For a patient diagnosed with **Flu** (severity: mild):

- Start node: `rest_prescribed`
- A* search finds: `rest_prescribed → treated` (cost 1)
- This is the lowest-cost path from that entry point to the goal

For a patient diagnosed with **Pneumonia** (severity: severe):

- Start node: `prescription_medication`
- A* search finds: `prescription_medication → treated` (cost 3) or
  `prescription_medication → specialist_referral → treated` (cost 7)
- A* selects the lower-cost path: `[prescription_medication, treated]`

If no path exists (disconnected subgraph), the agent falls back to:
`[start_node, "specialist_referral", "treated"]`

---

## 4. Constraint Satisfaction Problem (CSP)

**Agent:** `DrugSafetyAgent` (`agents/drug_safety_agent.py`)
**Knowledge base:** `knowledge/drug_interactions.py`

### 4.1 CSP Formulation

The drug safety check is modelled as a binary CSP:

- **Variables:** Each drug in the combined set (disease-specific drugs + any existing patient medications). For example, for Angina: `{Aspirin, Beta_Blockers, Nitroglycerin}`.
- **Domains:** Each variable has domain `{"safe", "flagged"}` — a drug is either safe to prescribe or flagged due to an interaction.
- **Constraints:** For every forbidden pair `(drug_a, drug_b)` in `FORBIDDEN_PAIRS`, a binary constraint is added: both drugs cannot simultaneously hold the value `"safe"`.

```python
# CSP construction
problem = Problem()
for drug in all_drugs:
    problem.addVariable(drug, ["safe", "flagged"])

for drug_a, drug_b in FORBIDDEN_PAIRS:
    if drug_a in all_drugs and drug_b in all_drugs:
        problem.addConstraint(
            lambda a, b: not (a == "safe" and b == "safe"),
            (drug_a, drug_b)
        )

solutions = problem.getSolutions()
```

### 4.2 Variables and Domains

```
Variables:  {drug_1, drug_2, ..., drug_n}
            where each drug ∈ DISEASE_DRUGS[disease] ∪ existing_drugs

Domains:    D(drug_i) = {"safe", "flagged"}

Constraints:
    ∀ (drug_a, drug_b) ∈ FORBIDDEN_PAIRS:
        ¬(drug_a == "safe" ∧ drug_b == "safe")
```

### 4.3 Forbidden Pairs (Hard Constraints)

```python
FORBIDDEN_PAIRS = [
    ("Aspirin",      "Warfarin"),
    ("Ibuprofen",    "Warfarin"),
    ("Metformin",    "Alcohol"),
    ("Beta_Blockers","Verapamil"),
    ("ACE_Inhibitors","Potassium_Supplements"),
]
```

Each pair represents a clinically dangerous co-prescription that the CSP is designed to detect and block.

### 4.4 Violation Detection and Reporting

The primary output is not the CSP solution set but the **violation list**: any forbidden pair where both drugs are present in the active drug set is recorded as an interaction regardless of CSP variable assignment, because the clinical risk exists whenever both drugs could be co-prescribed.

```
FUNCTION check_safety(disease, existing_drugs=[]):
    all_drugs ← DISEASE_DRUGS[disease] ∪ existing_drugs

    violations ← []
    alternatives ← {}

    FOR (drug_a, drug_b) IN FORBIDDEN_PAIRS:
        IF drug_a ∈ all_drugs AND drug_b ∈ all_drugs:
            violations.APPEND("{drug_a} + {drug_b} interaction detected")
            FOR d IN {drug_a, drug_b}:
                IF d ∈ DRUG_ALTERNATIVES:
                    alternatives[d] ← DRUG_ALTERNATIVES[d]

    RETURN {
        safe:         len(violations) == 0,
        drugs:        all_drugs,
        violations:   violations,
        alternatives: alternatives
    }
```

### 4.5 Alternative Drug Suggestions

When a violation is detected, `DRUG_ALTERNATIVES` is consulted to surface a safer substitute:

```python
DRUG_ALTERNATIVES = {
    "Aspirin":        "Acetaminophen",
    "Ibuprofen":      "Acetaminophen",
    "Warfarin":       "Dabigatran",
    "Metformin":      "Sitagliptin",
    "Beta_Blockers":  "Amlodipine",
    "ACE_Inhibitors": "Amlodipine",
}
```

The agent is designed to be stateless: each call to `check_safety` constructs an independent CSP instance. This means it is safe to share a single `DrugSafetyAgent` instance across concurrent consultation sessions without locking.

---

## 5. Multi-Agent Orchestration

**Agent:** `OrchestratorAgent` (`agents/orchestrator.py`)

### 5.1 State Machine Design

The OrchestratorAgent implements a six-state FSM using Python's `enum.Enum`. Each state corresponds to a well-defined phase of the diagnostic pipeline and maps to exactly one specialist agent call:

```python
class State(Enum):
    COLLECTING = auto()   # SymptomCollectionAgent
    DIAGNOSING = auto()   # DiagnosisAgent
    PLANNING   = auto()   # TreatmentPlannerAgent
    CHECKING   = auto()   # DrugSafetyAgent
    EXPLAINING = auto()   # ExplanationAgent
    DONE       = auto()   # Terminal state
```

### 5.2 Agent Hand-Off Protocol

The `process(user_input)` method is the single entry point. On each call it attempts to advance through as many states as possible in a single synchronous turn, stopping only when user input is required or when the pipeline reaches DONE.

```
FUNCTION process(user_input):
    IF state == DONE:
        RETURN "Session complete."

    IF state == COLLECTING:
        new_symptoms ← symptom_agent.extract_symptoms(user_input)
        symptoms ← sorted(set(symptoms ∪ new_symptoms))

        IF NOT symptom_agent.has_enough_symptoms(symptoms):
            RETURN symptom_agent.get_followup_question(symptoms)
                   # State stays COLLECTING — pause for user input

        state ← DIAGNOSING

    IF state == DIAGNOSING:
        diagnosis ← diagnosis_agent.diagnose(symptoms)

        IF diagnosis is empty OR diagnosis[0].confidence < 0.50:
            state ← COLLECTING              # Backtrack
            RETURN "Need more symptoms..."  # Pause for user input

        state ← PLANNING

    IF state == PLANNING:
        treatment ← treatment_agent.plan_treatment(diagnosis[0].disease)
        state ← CHECKING

    IF state == CHECKING:
        safety ← drug_agent.check_safety(diagnosis[0].disease)
        state ← EXPLAINING

    IF state == EXPLAINING:
        report ← explanation_agent.explain(symptoms, diagnosis, treatment, safety)
        state ← DONE
        RETURN report
```

### 5.3 Session Lifecycle

| Method | Effect |
|---|---|
| `__init__()` | Instantiates all five specialist agents; calls `reset()` |
| `reset()` | Sets state to COLLECTING; clears symptoms, diagnosis, treatment, safety |
| `process(user_input)` | Advances FSM; returns string (question or report) |

Each MediAgent Streamlit session creates one `OrchestratorAgent` instance. When the user clicks "New Chat", the UI replaces the existing agent instance with a freshly constructed one, ensuring complete state isolation between consultations.

### 5.4 Precondition Guarantees

The FSM design provides formal precondition guarantees for downstream agents:

- `DiagnosisAgent.diagnose()` is never called with fewer than 2 symptoms (MIN_SYMPTOMS gate in COLLECTING).
- `TreatmentPlannerAgent.plan_treatment()` is never called unless `diagnosis[0].confidence >= 0.50`.
- `DrugSafetyAgent.check_safety()` is never called unless a valid treatment pathway exists.
- `ExplanationAgent.explain()` is never called unless symptoms, diagnosis, treatment, and safety are all populated.

These guarantees eliminate an entire class of runtime errors where downstream agents would receive empty or inconsistent inputs.
