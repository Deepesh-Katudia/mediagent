# MediAgent — Multi-Agent Healthcare Diagnostic Assistant

A multi-agent AI system that diagnoses medical conditions from symptoms and generates safe treatment plans.

## Team
- [Add team members]

## Setup
```bash
pip install -r requirements.txt
cd mediagent && python ml/train.py
streamlit run ui/app.py
```

## Agents
- **OrchestratorAgent** — routes conversation state
- **SymptomCollectionAgent** — symptom extraction and normalization
- **DiagnosisAgent** — FOL inference + Naive Bayes classification
- **TreatmentPlannerAgent** — A* treatment path search
- **DrugSafetyAgent** — CSP drug interaction checker
- **ExplanationAgent** — natural language summary

## Testing
```bash
pytest --cov
```
