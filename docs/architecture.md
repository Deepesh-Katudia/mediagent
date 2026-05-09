# MediAgent — System Architecture

## 1. System Overview

MediAgent is a multi-agent healthcare diagnostic assistant built as an academic demonstration of classical and statistical AI techniques applied to clinical reasoning. The system accepts free-text symptom descriptions from a patient via a Streamlit chat interface, routes the input through a pipeline of six specialised agents, and returns a structured diagnostic report that includes a ranked differential diagnosis, an optimal treatment pathway, a drug safety check, and a plain-English explanation.

**Key capabilities:**

- Natural-language symptom extraction with synonym normalisation
- Hybrid symbolic + probabilistic disease diagnosis
- Cost-optimal treatment pathway planning
- Drug interaction detection with alternative recommendations
- Session history management with multi-consultation support

MediAgent is explicitly an educational prototype. It must not be used as a substitute for professional medical advice.

---

## 2. Agent Architecture

| Agent | Role | AI Technique | Key Module |
|---|---|---|---|
| **OrchestratorAgent** | Top-level pipeline controller; drives the finite state machine (FSM) that sequences all specialist agents and manages inter-agent hand-offs | Finite State Machine (FSM) | `agents/orchestrator.py` |
| **SymptomCollectionAgent** | Parses free-text patient input; maps colloquial phrases and synonyms to canonical symptom tokens; generates contextual follow-up questions when the symptom set is insufficient | Rule-based NLP with synonym mapping | `agents/symptom_collector.py` |
| **DiagnosisAgent** | Produces a ranked list of candidate diagnoses by blending First-Order Logic forward chaining with a pre-trained Naive Bayes classifier | FOL + Forward Chaining, Naive Bayes (sklearn) | `agents/diagnosis_agent.py` |
| **TreatmentPlannerAgent** | Finds the minimum-cost care pathway through a weighted directed treatment graph using A* heuristic search | A* Search (NetworkX) | `agents/treatment_planner.py` |
| **DrugSafetyAgent** | Checks whether the drugs associated with the top diagnosis contain any dangerous pairwise interactions; suggests safer alternatives when conflicts are detected | Constraint Satisfaction Problem (python-constraint) | `agents/drug_safety_agent.py` |
| **ExplanationAgent** | Formats all upstream agent outputs — diagnosis, treatment pathway, drug safety report — into a single coherent Markdown narrative for the user | Template-driven NLG | `agents/explanation_agent.py` |

---

## 3. Architecture Diagram

The following ASCII diagram shows the top-level data flow through the system. The OrchestratorAgent is the single entry point; it delegates to specialist agents in strict FSM order and aggregates their outputs before forwarding the final report to the user.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI  (ui/app.py)                       │
│  ┌──────────────┐                                  ┌─────────────────┐  │
│  │  Chat Input  │──── free-text prompt ──────────▶ │  Chat Output    │  │
│  └──────────────┘                                  └────────▲────────┘  │
└────────────────────────────────────────────────────────────│────────────┘
                                │                            │
                                ▼                            │
         ┌──────────────────────────────────────────────┐    │
         │              OrchestratorAgent                │    │
         │          (Finite State Machine)               │────┘
         │   COLLECTING → DIAGNOSING → PLANNING          │
         │      → CHECKING → EXPLAINING → DONE           │
         └───────────────────┬──────────────────────────-┘
                             │
          ┌──────────────────┼──────────────────────────┐
          │                  │                          │
          ▼                  ▼                          ▼
┌──────────────────┐ ┌───────────────┐      ┌─────────────────────┐
│ SymptomCollection│ │ DiagnosisAgent│      │ TreatmentPlannerAgent│
│     Agent        │ │ FOL + Naive   │      │    A* Search on      │
│  (Rule-based NLP)│ │     Bayes     │      │  Treatment Graph     │
└────────┬─────────┘ └───────┬───────┘      └──────────┬──────────┘
         │                   │                         │
         │ canonical         │ ranked                  │ ordered
         │ symptom list      │ diagnoses               │ pathway
         │                   │                         │
         └───────────────────┴──────────────┬──────────┘
                                            │
                                ┌───────────┼───────────┐
                                ▼                       ▼
                     ┌─────────────────┐    ┌──────────────────────┐
                     │  DrugSafetyAgent│    │   ExplanationAgent   │
                     │  CSP Interaction│    │  Template-driven NLG │
                     │    Checking     │    │  (Final Report)      │
                     └────────┬────────┘    └──────────┬───────────┘
                              │                        │
                              └───────── safety ───────┘
                                          report
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │  User receives  │
                                   │ diagnostic report│
                                   └─────────────────┘
```

---

## 4. OrchestratorAgent State Machine

The OrchestratorAgent implements a six-state finite state machine. States progress linearly from `COLLECTING` to `DONE`. The only permitted backward transition is from `DIAGNOSING` back to `COLLECTING` when diagnostic confidence falls below the 0.50 threshold — this forces the agent to request additional symptoms before attempting another inference pass.

```
                    ┌─────────────────────────────────────┐
                    │  Low confidence (< 0.50)             │
                    │  "Need more symptoms"                │
                    │                                     │
          ┌─────────▼─────────┐                           │
  START──▶│   COLLECTING      │                           │
          │  extract_symptoms  │                           │
          │  has_enough?       │                           │
          └────────┬──────────┘                           │
                   │ >= MIN_SYMPTOMS (2)                   │
                   ▼                                       │
          ┌────────────────────┐                          │
          │    DIAGNOSING      │──────────────────────────┘
          │  FOL + Naive Bayes │
          │  confidence >= 0.5?│
          └────────┬───────────┘
                   │ yes
                   ▼
          ┌────────────────────┐
          │     PLANNING       │
          │   A* Treatment     │
          │      Search        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │     CHECKING       │
          │   CSP Drug Safety  │
          │       Check        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │    EXPLAINING      │
          │  Format Markdown   │
          │      Report        │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │       DONE         │
          │  Session complete  │
          └────────────────────┘
```

**State transition rules:**

| From | To | Condition |
|---|---|---|
| COLLECTING | DIAGNOSING | `len(symptoms) >= 2` |
| DIAGNOSING | PLANNING | `top_diagnosis.confidence >= 0.50` |
| DIAGNOSING | COLLECTING | `top_diagnosis.confidence < 0.50` (backtrack) |
| PLANNING | CHECKING | Always (A* completes or falls back) |
| CHECKING | EXPLAINING | Always (CSP completes) |
| EXPLAINING | DONE | Always (report formatted) |

---

## 5. Component Dependencies

```
ui/app.py
  └── agents/orchestrator.py
        ├── agents/symptom_collector.py
        │     └── knowledge/disease_rules.py  (SYMPTOMS)
        ├── agents/diagnosis_agent.py
        │     ├── knowledge/disease_rules.py  (DISEASE_RULES, SYMPTOMS)
        │     └── ml/model.pkl                (MultinomialNB, loaded via joblib)
        ├── agents/treatment_planner.py
        │     └── knowledge/treatment_graph.py
        │           (DISEASE_SEVERITY, SEVERITY_START, TREATMENT_HEURISTIC,
        │            build_treatment_graph → NetworkX DiGraph)
        ├── agents/drug_safety_agent.py
        │     └── knowledge/drug_interactions.py
        │           (DISEASE_DRUGS, FORBIDDEN_PAIRS, DRUG_ALTERNATIVES)
        └── agents/explanation_agent.py
              (no external knowledge imports — pure formatter)

ml/train.py
  ├── ml/dataset_generator.py
  │     └── knowledge/disease_rules.py
  └── knowledge/disease_rules.py
        → output: ml/model.pkl
```

The knowledge base modules (`knowledge/`) are pure data dictionaries with no cross-dependencies between them. All reasoning logic lives in `agents/`. The `ml/` package handles offline training; at runtime, only `ml/model.pkl` is consumed.

---

## 6. Design Decisions

### Why Multi-Agent Architecture?

The multi-agent design enforces separation of concerns at the architectural level. Each agent owns a single, well-defined responsibility:

- **Testability:** Each agent can be unit-tested in isolation (see `tests/`). The orchestrator is tested by mocking the five specialists.
- **Replaceability:** Any agent can be swapped for a more sophisticated implementation (e.g., replacing the Naive Bayes classifier with a transformer-based model) without touching the others.
- **Auditability:** The FSM produces a deterministic, traceable execution path. Every state transition is logged implicitly by the agent's return value, making the diagnostic reasoning transparent to the user and to developers.
- **Academic clarity:** Each agent maps cleanly to a distinct AI technique from Russell and Norvig, making the system suitable as a teaching artefact.

### Why These Specific AI Techniques?

| Technique | Rationale |
|---|---|
| **FOL + Forward Chaining** | Encodes expert medical knowledge as explicit, interpretable rules. Required-symptom gates prevent spurious diagnoses and give the system clinically defensible hard constraints. |
| **Naive Bayes** | Complements the symbolic layer with data-driven probability estimates. The independence assumption is acceptable for a demonstration system; the synthetic training data preserves the distributional shape of the rule base. |
| **Hybrid Blending (60/40)** | The FOL layer is weighted more heavily (0.6) because it embeds domain expertise and provides hard eligibility gates. The ML layer (0.4) acts as a calibration signal to handle cases where symptom coverage is partial. |
| **A* Search** | A directed graph with edge costs naturally models treatment escalation. A* is preferred over BFS because the `TREATMENT_HEURISTIC` table encodes domain knowledge about expected remaining steps, reducing unnecessary exploration. |
| **CSP** | Drug interactions are inherently binary constraints (drug A and drug B must not both be prescribed). CSP formalises this structure cleanly and is extensible: adding a new forbidden pair requires only one line in `FORBIDDEN_PAIRS`. |
| **FSM (Orchestrator)** | A finite state machine gives a provably correct sequencing guarantee. The pipeline cannot skip stages, preventing downstream agents from being invoked with incomplete or inconsistent inputs. |
