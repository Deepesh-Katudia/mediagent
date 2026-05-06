# MediAgent — Multi-Agent Healthcare Diagnostic Assistant

A graduate AI project demonstrating multi-agent system design using classical AI techniques from Russell & Norvig's *Artificial Intelligence: A Modern Approach*.

## Team
- [Add team member names here]

## AI Techniques Demonstrated

| Technique | Agent | Chapter |
|---|---|---|
| PEAS Agent Architecture | OrchestratorAgent (utility-based) | Ch. 1-2 |
| First-Order Logic + Forward Chaining | DiagnosisAgent | Ch. 7-9 |
| Naive Bayes ML Classifier | DiagnosisAgent | Ch. 20 |
| A* Search | TreatmentPlannerAgent | Ch. 3-4 |
| Constraint Satisfaction (CSP) | DrugSafetyAgent | Ch. 5 |
| Multi-Agent Orchestration | OrchestratorAgent state machine | Ch. 1-2 |

## System Architecture

```
User symptoms
    ↓
OrchestratorAgent (state machine)
    ↓
SymptomCollectionAgent  →  normalizes symptoms
    ↓
DiagnosisAgent          →  FOL rules + Naive Bayes → top-3 diagnoses
    ↓
TreatmentPlannerAgent   →  A* search → optimal treatment path
    ↓
DrugSafetyAgent         →  CSP → drug interaction check
    ↓
ExplanationAgent        →  plain-English summary → Streamlit UI
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model (generates ml/model.pkl)
cd mediagent
python ml/train.py

# 3. Launch the app
streamlit run ui/app.py
```

## Testing

```bash
cd mediagent
pytest
```

Coverage target: ≥ 80% (enforced by pytest.ini)

## Disease Coverage (10 conditions)

| Category | Diseases |
|---|---|
| Respiratory | Flu, Common Cold, Pneumonia, Bronchitis |
| Cardiovascular | Hypertension, Angina |
| Metabolic | Type 2 Diabetes, Hypothyroidism |
| Neurological | Migraine, Tension Headache |

## File Structure

```
mediagent/
├── agents/           # 6 specialist agents + orchestrator
├── knowledge/        # FOL disease rules, treatment graph, drug interactions
├── ml/               # Synthetic dataset generator + Naive Bayes training
├── ui/               # Streamlit chat interface
└── tests/            # Unit tests + integration tests (60 tests)
```

## References

- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*, 4th ed.
- scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- NetworkX A* documentation: [https://networkx.org](https://networkx.org)
- python-constraint: [https://labix.org/python-constraint](https://labix.org/python-constraint)
