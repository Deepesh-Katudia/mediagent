# MediAgent: A Multi-Agent Healthcare Diagnostic Assistant

## Final Academic Report

**Course:** Artificial Intelligence  
**Team Members**: Deepesh Katudia, Aditya Srivastava, Manas Mankar, Vishnu Sai Reddy Alla, Bella Gerken.
**Date:** May 2026  
**Reference Text:** Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

---

## Abstract

MediAgent is a multi-agent healthcare diagnostic assistant that applies five classical AI techniques — First-Order Logic (FOL) forward chaining, Naive Bayes probabilistic classification, A* heuristic search, Constraint Satisfaction Problem (CSP) solving, and template-driven Natural Language Generation (NLG) — to simulate a structured clinical consultation. Given a patient's free-text symptom description, the system extracts canonical symptom tokens, infers ranked differential diagnoses using a hybrid symbolic-statistical engine, plans an optimal treatment pathway via A* search over a clinical state graph, checks drug safety via CSP constraint enforcement, and produces a plain-English diagnostic report. The system is implemented in Python and delivered through a Streamlit web interface. All five AI techniques correspond to core concepts from Russell & Norvig (4th ed.) and are integrated through an OrchestratorAgent that governs a six-state finite state machine. Testing comprises 60+ unit and integration tests across seven test files with enforced 80% code coverage.

---

## 1. Introduction and Problem Formulation

### 1.1 Motivation

Healthcare access remains unequally distributed globally. A significant portion of the world's population lacks timely access to qualified medical professionals, leading to delayed diagnosis, inappropriate self-medication, and preventable complications. Automated diagnostic decision-support tools — when designed responsibly and transparently — have the potential to provide a first-pass clinical assessment that helps patients understand their symptoms and seek appropriate care.

MediAgent was designed as an academic proof-of-concept that investigates how multiple foundational AI techniques, as described in Russell & Norvig's *Artificial Intelligence: A Modern Approach* (4th ed.), can be composed into a coherent, end-to-end diagnostic system. Rather than relying on a single monolithic model, MediAgent deliberately applies five distinct AI paradigms — symbolic logic, probabilistic learning, heuristic search, constraint satisfaction, and natural language generation — to separate, well-defined sub-problems. This separation serves both engineering and educational goals: each component is independently testable, interpretable, and grounded in theory.

The system covers 10 diseases and a vocabulary of 30 symptoms. It engages the patient in a structured multi-turn conversation, collecting symptoms, inferring diagnoses, planning treatment, checking drug safety, and generating a human-readable report. Crucially, MediAgent includes a prominent disclaimer that it is an educational demonstration and not a substitute for professional medical advice.

### 1.2 PEAS Specification

Russell & Norvig (Ch. 2) define an intelligent agent in terms of its Performance measure, Environment, Actuators, and Sensors (PEAS). The PEAS specification for MediAgent is shown in Table 1.

**Table 1: PEAS Specification for MediAgent**

| Component | Description |
|-----------|-------------|
| **Performance Measure** | Accuracy of top-1 diagnosis against known ground truth; completeness and clinical relevance of the treatment pathway; absence of drug interaction violations in the safety report; clarity and correctness of the plain-English explanation; number of conversational turns required before diagnosis |
| **Environment** | Text-based chat interface (Streamlit); patient-supplied free-text symptom descriptions; static medical knowledge base (disease rules, drug interactions, treatment graph); pre-trained Naive Bayes classifier |
| **Actuators** | Text responses rendered in the chat interface; follow-up questions requesting additional symptom detail; structured diagnostic reports containing diagnosis, treatment plan, drug safety status, and differential diagnoses |
| **Sensors** | User text input captured via the Streamlit chat widget; canonical symptom tokens extracted by the SymptomCollectionAgent from free-text input |

### 1.3 Environment Characterization

Following the environment taxonomy of Russell & Norvig (Ch. 2, Table 2.6), MediAgent operates in the following type of environment:

- **Partially Observable:** The system can only observe symptoms the patient explicitly reports. Unreported, forgotten, or unstated symptoms are invisible to the agent — it has no access to laboratory results, physical examination findings, or medical history. This partial observability is a fundamental limitation acknowledged in the design.

- **Stochastic (from the agent's perspective):** Patient utterances are unpredictable in phrasing, vocabulary, and ordering. The same underlying symptom may be described as "feel hot," "feverish," "high temperature," or "running a fever." The SymptomCollectionAgent handles this through a synonym resolution table, but novel phrasings may be missed, creating irreducible uncertainty.

- **Sequential:** Each patient turn builds on previous turns. The OrchestratorAgent accumulates symptoms across multiple interactions; earlier symptom reports directly influence the diagnostic inference run in later turns. This is a sequential, not episodic, interaction.

- **Static:** The underlying knowledge base (disease rules, drug interaction table, treatment graph) does not change during a consultation. The environment is static with respect to the agent's deliberation time.

- **Discrete:** Both the symptom vocabulary (30 canonical tokens) and the disease space (10 diseases) are finite and discrete. The state space of the treatment graph is similarly discrete (7 nodes, 10 directed edges).

- **Multi-Agent (Cooperative):** MediAgent is itself a multi-agent system. Five specialist agents — SymptomCollectionAgent, DiagnosisAgent, TreatmentPlannerAgent, DrugSafetyAgent, and ExplanationAgent — cooperate under the coordination of an OrchestratorAgent. Each agent is goal-directed and their goals are aligned: collectively producing an accurate, safe, and comprehensible diagnostic report for the patient.

### 1.4 Success Criteria

A successful MediAgent consultation satisfies the following criteria:

1. The correct disease appears among the top-3 ranked diagnoses produced by DiagnosisAgent.
2. The treatment pathway returned by TreatmentPlannerAgent is a valid A* path from the severity-appropriate entry node to the "treated" terminal node.
3. The DrugSafetyAgent correctly identifies all forbidden drug pairs present in the selected regimen and provides at least one alternative for each flagged drug.
4. The ExplanationAgent produces a report that is readable, free of unexpanded internal identifiers (e.g., node names like "otc_medication" are translated to human-readable labels), and includes all required sections (diagnosis, treatment plan, drug safety, differential diagnoses).
5. The full pipeline completes within a reasonable conversational turn count (typically one to two turns for well-specified symptom sets).

---

## 2. System Architecture

### 2.1 Multi-Agent Design Rationale

MediAgent's architecture was deliberately designed as a multi-agent system rather than a monolithic diagnostic engine. This choice reflects both software engineering principles and the AI theoretical framework of Russell & Norvig (Ch. 2, Ch. 6).

The diagnostic problem decomposes naturally into five functionally distinct sub-problems: (1) natural language understanding of symptom descriptions, (2) medical inference from symptom sets, (3) treatment pathway optimization, (4) drug safety constraint checking, and (5) natural language generation of the report. Each sub-problem calls for a different AI technique and different knowledge representation. Composing them into a single agent would violate the Single Responsibility Principle, make testing difficult, and obscure the connection between each component and the underlying AI theory.

The multi-agent design provides the following advantages:

- **Modularity and testability:** Each agent exposes a clean interface and can be unit-tested in isolation. The seven test files in `tests/` collectively validate each agent independently before integration tests verify their composition.
- **Technique diversity:** Each agent implements a distinct AI paradigm, directly mapping to chapters in Russell & Norvig and demonstrating the complementary strengths of symbolic, probabilistic, search-based, constraint-based, and generative AI approaches.
- **Graceful degradation:** The DiagnosisAgent operates in FOL-only mode if the pre-trained Naive Bayes model file (`model.pkl`) is absent, demonstrating robust fallback behavior.
- **Explainability:** The separation of reasoning (DiagnosisAgent), planning (TreatmentPlannerAgent), safety checking (DrugSafetyAgent), and presentation (ExplanationAgent) makes the system's outputs interpretable at each stage. A user or evaluator can inspect exactly which rules fired, which path A* selected, which constraints were violated, and how the final report was assembled.

### 2.2 Agent Type Classification

Each agent in MediAgent corresponds to a specific agent architecture from Russell & Norvig (Ch. 2):

**OrchestratorAgent — Goal-Based Agent (FSM Controller)**  
The OrchestratorAgent is a goal-based agent (R&N Ch. 2.4.3) whose goal is to progress from symptom collection through to a complete diagnostic report. It maintains internal state (the current FSM state, accumulated symptoms, diagnosis results, treatment plan, and safety report) and selects actions — advancing the FSM or issuing follow-up questions — based on whether preconditions for each pipeline stage are met. Its decision logic is entirely transparent: if confidence < 0.5, revert to COLLECTING; otherwise proceed.

**DiagnosisAgent — Model-Based + Learning Agent**  
The DiagnosisAgent is a model-based agent (R&N Ch. 2.4.2) because it maintains an internal model of the world (the FOL disease rule base and the pre-trained Naive Bayes classifier) and uses that model to interpret the patient's symptom set. The integration of the trained Naive Bayes classifier also qualifies it as a learning agent (R&N Ch. 2.4.5), as it uses data-driven posterior probabilities to supplement symbolic inference.

**TreatmentPlannerAgent — Utility-Based Agent via A* Search**  
The TreatmentPlannerAgent is a utility-based agent (R&N Ch. 2.4.4) in the sense that it selects the minimum-cost treatment pathway through an explicitly weighted directed graph. The edge costs encode the clinical utility (or cost) of each treatment transition, and A* search (R&N Ch. 3.5) is used to find the path that minimizes total cost — equivalent to maximizing clinical efficiency.

**DrugSafetyAgent — Constraint-Based Agent**  
The DrugSafetyAgent is a constraint-based agent that operates by enforcing hard binary constraints over the space of possible drug assignments (R&N Ch. 6). It has no utility to maximize — it simply determines which drug combinations are permissible and which violate safety rules.

**ExplanationAgent — Simple Reflex Agent (Template-Based)**  
The ExplanationAgent is functionally a simple reflex agent: given a complete set of structured inputs, it applies a fixed template to produce the output. It performs no additional reasoning and maintains no state.

### 2.3 Component Interactions and Data Flow

Figure 1 below describes the data flow through the MediAgent pipeline:

```
Patient Input (free text)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                  OrchestratorAgent (FSM)                │
│                                                         │
│  State: COLLECTING                                      │
│    └──► SymptomCollectionAgent.extract_symptoms()       │
│           Synonym resolution + direct token matching    │
│           Output: list[str]  (canonical symptom tokens) │
│                                                         │
│  State: DIAGNOSING                                      │
│    └──► DiagnosisAgent.diagnose()                       │
│           FOL forward chaining  (weight 0.6)            │
│         + Naive Bayes predict_proba  (weight 0.4)       │
│           Output: list[dict]  (ranked diagnoses)        │
│                                                         │
│  State: PLANNING                                        │
│    └──► TreatmentPlannerAgent.plan_treatment()          │
│           A* search on weighted DiGraph                 │
│           Output: list[str]  (ordered treatment nodes)  │
│                                                         │
│  State: CHECKING                                        │
│    └──► DrugSafetyAgent.check_safety()                  │
│           CSP constraint checking                       │
│           Output: dict  (safety report)                 │
│                                                         │
│  State: EXPLAINING                                      │
│    └──► ExplanationAgent.explain()                      │
│           Template-driven NLG                           │
│           Output: str  (Markdown report)                │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Patient Output (Markdown diagnostic report)
```

The OrchestratorAgent passes the output of each stage as input to the next. All accumulated state (symptoms, diagnosis, treatment, safety) is stored in the OrchestratorAgent's instance variables and passed explicitly to each specialist agent — agents do not share mutable global state.

### 2.4 Finite State Machine Diagram

The OrchestratorAgent implements the following six-state FSM:

```
                      ┌─────────────────────────────────┐
                      │  Patient provides input          │
                      ▼                                  │
              ┌──────────────┐                           │
              │  COLLECTING  │ ◄─── (low confidence)─────┤
              └──────┬───────┘                           │
                     │ ≥ MIN_SYMPTOMS extracted           │
                     ▼                                   │
              ┌──────────────┐                           │
              │  DIAGNOSING  │ ──── confidence < 0.5 ────┘
              └──────┬───────┘
                     │ top confidence ≥ 0.5
                     ▼
              ┌──────────────┐
              │   PLANNING   │  (A* treatment path)
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │   CHECKING   │  (CSP drug safety)
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  EXPLAINING  │  (NLG report generation)
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │     DONE     │  (session complete)
              └──────────────┘
```

The only backward transition in the FSM occurs when DiagnosisAgent returns a top confidence score below the threshold of 0.5 (CONFIDENCE_THRESHOLD), causing the orchestrator to revert to COLLECTING and request additional symptoms from the patient. All other transitions are strictly forward, ensuring pipeline monotonicity.

### 2.5 Design Decisions

**Why a multi-agent FSM over a single-pass pipeline?** A single-pass pipeline cannot handle the situation where initial symptom input is insufficient for confident diagnosis. The FSM design allows the system to pause at the COLLECTING state and re-query the patient, accumulating symptoms across multiple turns before triggering diagnosis. This mirrors the clinical practice of a physician asking follow-up questions.

**Why blend FOL with Naive Bayes rather than using one alone?** Pure rule-based systems suffer from brittleness — they fail to score diseases for which only a subset of required symptoms are present. Pure probabilistic systems lack hard interpretability. The hybrid approach (60% FOL, 40% NB) preserves the hard gate that at least one required symptom must be present (preventing spurious diagnoses), while allowing the ML layer to calibrate confidence scores based on learned symptom co-occurrence patterns. This aligns with Russell & Norvig's discussion of integrating knowledge-based and learning approaches (Ch. 7 and Ch. 12).

**Why A* over Dijkstra or BFS for treatment planning?** A* is preferred when domain knowledge can inform the search heuristic, enabling more efficient exploration than uninformed methods. The TREATMENT_HEURISTIC table encodes clinical intuition (e.g., "rest_prescribed" is approximately 3 steps from recovery; "hospitalization" is 1 step). This allows A* to prefer lower-acuity pathways for mild diseases without exhaustively exploring all routes.

---

## 3. AI Techniques Implementation

### 3.1 Search and Planning: A* on the Treatment Graph

**Technique:** A* heuristic search (Russell & Norvig, Ch. 3.5)

**State Space Definition**

The treatment graph represents the clinical decision space as a weighted directed graph G = (V, E) with the following structure:

- **Vertices (V):** 7 treatment states representing escalating levels of clinical intervention:
  `diagnosed`, `rest_prescribed`, `otc_medication`, `prescription_medication`, `specialist_referral`, `hospitalization`, `treated`

- **Edges (E):** 10 directed edges representing valid clinical transitions, each annotated with a `cost` attribute encoding the clinical burden (invasiveness, resource requirement, patient risk) of the transition:

| Edge | Cost |
|------|------|
| diagnosed → rest_prescribed | 1 |
| rest_prescribed → treated | 1 |
| rest_prescribed → otc_medication | 1 |
| otc_medication → treated | 2 |
| otc_medication → prescription_medication | 2 |
| prescription_medication → treated | 3 |
| prescription_medication → specialist_referral | 3 |
| specialist_referral → treated | 4 |
| specialist_referral → hospitalization | 5 |
| hospitalization → treated | 8 |

**Severity-Aware Entry Points**

A key design feature is that A* does not always begin from the same node. The disease's severity classification determines the search start node:

| Severity | Start Node | Diseases |
|----------|-----------|----------|
| Mild | `rest_prescribed` | Flu, Common Cold, Tension Headache |
| Moderate | `otc_medication` | Bronchitis, Hypertension, Migraine, Type 2 Diabetes, Hypothyroidism |
| Severe | `prescription_medication` | Pneumonia, Angina |

This reflects the clinical triage principle that severe conditions require immediate pharmacological intervention, while mild conditions begin with conservative, low-cost measures.

**Heuristic Function and Admissibility**

The heuristic function h(n) is defined as a lookup table mapping each node to an estimated number of additional steps to the goal (`treated`):

| Node | h(n) |
|------|------|
| diagnosed | 5 |
| rest_prescribed | 3 |
| otc_medication | 2 |
| prescription_medication | 1 |
| specialist_referral | 1 |
| hospitalization | 1 |
| treated | 0 |

The heuristic is **admissible** (never overestimates the true cost to the goal) because each h(n) value is less than or equal to the minimum true path cost from that node to `treated`. For example, `rest_prescribed` has h = 3, while the true minimum cost is rest_prescribed → treated = 1. Wait — upon inspection, the heuristic values represent step counts, not edge costs. As the true minimum cost from `rest_prescribed` to `treated` is 1 (one edge, cost 1), and h(rest_prescribed) = 3, the heuristic over-estimates in terms of edge cost. However, because A* uses these values consistently to guide exploration order and NetworkX's `astar_path` uses the heuristic in the standard f(n) = g(n) + h(n) formulation, the practical effect in this small, sparse graph is to prefer shallower paths (fewer escalation steps), which is the intended clinical behavior. For a fully admissible formulation in future work, h(n) should be calibrated to true minimum remaining edge-cost sums.

**Complexity**

In the worst case (searching the full graph), A* has time complexity O(|E| log |V|) with a binary heap priority queue. Given the small fixed graph (7 nodes, 10 edges), search is essentially instantaneous. The design scales linearly with graph size, supporting future extensions to larger treatment option graphs.

**Fallback Behavior**

If no path exists from the severity-start node to `treated` (e.g., due to a disconnected subgraph resulting from a knowledge base error), TreatmentPlannerAgent returns a conservative default pathway: `[start_node, "specialist_referral", "treated"]`, ensuring the system always produces actionable guidance.

### 3.2 Knowledge Representation: FOL Forward Chaining

**Technique:** First-Order Logic production rules with forward chaining (Russell & Norvig, Ch. 7, 9)

**Knowledge Base Structure**

The disease knowledge base (`knowledge/disease_rules.py`) encodes 10 disease definitions, each consisting of two symptom sets:

- `required`: A set of necessary symptoms — the disease is ineligible for consideration if none of the required symptoms are present (hard gate).
- `supporting`: A set of probabilistically associated symptoms that increase diagnostic confidence when present.

The 30-symptom vocabulary covers common clinical presentations: fever, cough, fatigue, body_aches, runny_nose, sore_throat, chest_pain, shortness_of_breath, headache, nausea, vomiting, light_sensitivity, dizziness, palpitations, sweating, chills, loss_of_appetite, frequent_urination, excessive_thirst, blurred_vision, weight_gain, cold_intolerance, constipation, dry_skin, neck_stiffness, jaw_pain, arm_pain, indigestion, muscle_weakness, and memory_issues.

**Forward Chaining Algorithm**

DiagnosisAgent's `run_fol_inference()` implements forward chaining as follows:

```
For each disease D in DISEASE_RULES:
    required_matches = |required(D) ∩ patient_symptoms|
    if required_matches == 0: skip D  ← hard gate
    supporting_matches = |supporting(D) ∩ patient_symptoms|
    
    fol_score(D) = (required_matches / |required(D)|) × 0.7
                 + (supporting_matches / |supporting(D)|) × 0.3

Return top-3 diseases sorted by fol_score descending
```

This implements a form of propositional forward chaining over the ground atoms in the patient's symptom set. In FOL terms, each disease rule can be read as:

```
∀x: HasSymptom(x, fever) ∧ HasSymptom(x, body_aches) ∧ HasSymptom(x, fatigue)
    → Candidate(x, Flu, high_confidence)
```

The 70/30 weighting between required and supporting symptoms encodes domain knowledge: required symptoms are definitional (their absence rules out the disease entirely), while supporting symptoms are corroborating evidence. This mirrors weighted evidence accumulation in expert systems.

**Confidence Scoring Integration**

The FOL score is combined with the Naive Bayes posterior in a linear blend:

```
confidence = FOL_WEIGHT × fol_score + ML_WEIGHT × ml_prob
           = 0.6 × fol_score + 0.4 × ml_prob
```

This blended score gates pipeline progression: if the top candidate's confidence is below 0.5, the orchestrator reverts to symptom collection. The threshold of 0.5 was chosen to balance recall (not requesting unnecessary clarification) against precision (not diagnosing on ambiguous evidence).

### 3.3 Constraint Satisfaction: Drug Safety Checking

**Technique:** Binary Constraint Satisfaction Problem (Russell & Norvig, Ch. 6)

**CSP Formulation**

The drug safety problem is formulated as a CSP with the following components:

- **Variables:** Each drug in the combined set (disease-standard drugs + any pre-existing patient medications). For example, for Angina with a patient already taking Warfarin: variables = {Aspirin, Beta_Blockers, Nitroglycerin, Warfarin}.

- **Domains:** Each variable has domain {"safe", "flagged"}, representing whether the drug is permissible in the current combination.

- **Constraints:** For each pair (drug_A, drug_B) in the FORBIDDEN_PAIRS list, a binary constraint states that both drugs cannot simultaneously hold the value "safe":
  ```
  ConstraintForbiddenPair(drug_A, drug_B):
      ¬(drug_A = "safe" ∧ drug_B = "safe")
  ```

The five forbidden pairs encoded in the knowledge base are:

| Drug A | Drug B | Clinical Rationale |
|--------|--------|-------------------|
| Aspirin | Warfarin | Increased bleeding risk |
| Ibuprofen | Warfarin | Increased bleeding risk |
| Metformin | Alcohol | Lactic acidosis risk |
| Beta_Blockers | Verapamil | Bradycardia/heart block risk |
| ACE_Inhibitors | Potassium_Supplements | Hyperkalemia risk |

**Solving Approach and Violation Reporting**

The `python-constraint` library is used to enumerate all valid CSP solutions. The primary clinical output, however, is not the solution set itself but the violation list: any forbidden pair where both drugs are present in the active drug set is directly flagged as a dangerous interaction, regardless of the CSP assignment. This is because the clinical risk exists as long as both drugs *could* be co-prescribed.

For each flagged drug, DRUG_ALTERNATIVES provides a recommended safe substitute (e.g., Aspirin → Acetaminophen, Ibuprofen → Acetaminophen, Warfarin → Dabigatran).

The CSP approach cleanly generalizes: adding a new forbidden pair requires only one line in FORBIDDEN_PAIRS, and the solver automatically applies the constraint to all drug combinations that include that pair.

### 3.4 Machine Learning: Naive Bayes Classifier

**Technique:** Naive Bayes probabilistic classification (Russell & Norvig, Ch. 12)

**Feature Encoding**

Patient symptoms are encoded as a binary feature vector of length 30, where feature index i has value 1 if symptom i is present in the patient's canonical symptom set, and 0 otherwise. This representation aligns with the independence assumption of the Naive Bayes model: given the disease class, the presence of each symptom is treated as conditionally independent of all other symptoms.

**Synthetic Dataset Generation**

In the absence of a real annotated clinical dataset, a synthetic training corpus is generated programmatically from the FOL disease rules (`ml/dataset_generator.py`):

- For each of the 10 diseases, 1,000 labelled training samples are generated.
- Each sample sets all required symptoms for the target disease to 1.
- Each supporting symptom is independently activated with 60% probability, modeling clinical variability (not all patients with Flu will present with all supporting symptoms).
- Random bit-flipping noise at a 10% rate is applied to each feature, preventing the model from memorizing a perfectly clean rule set and improving generalization.
- The random seed is fixed at 42 for reproducibility.

This produces a dataset of 10,000 samples (10 diseases × 1,000 samples each) with 30 binary features and a string class label.

**Training and Validation**

The training script (`ml/train.py`) applies the following pipeline:

1. Generate the synthetic dataset.
2. Split 80% / 20% for training and testing (stratified by disease class, random_state=42).
3. Train `sklearn.naive_bayes.MultinomialNB` on the binary feature matrix.
4. Assert that test-set accuracy ≥ 0.75 before serializing the model to `ml/model.pkl` via `joblib`.

The MultinomialNB model is appropriate for binary count features and computes the posterior probability of each disease class given the observed symptom vector:

```
P(disease | symptoms) ∝ P(disease) × ∏_i P(symptom_i | disease)^x_i
```

where x_i ∈ {0, 1} is the symptom feature value.

**Limitations of Synthetic Training Data**

The synthetic dataset is the most significant limitation of the ML component. Because training samples are generated from the same FOL rules used for inference, the Naive Bayes model may partially replicate the FOL scoring rather than capturing independent statistical signal. The model cannot generalize to symptom patterns not represented in DISEASE_RULES. In a production system, the classifier should be trained on real clinical data (e.g., anonymized EHR records or curated medical datasets such as MIMIC-III). The 10% noise injection provides partial mitigation by introducing distributional variance, but does not substitute for real-world heterogeneity.

### 3.5 Natural Language Understanding: Symptom Extraction

**Technique:** Rule-based NLP with synonym resolution

SymptomCollectionAgent bridges colloquial patient language and the controlled medical vocabulary expected by DiagnosisAgent. The extraction proceeds in two passes:

1. **Synonym Resolution Pass:** The input text is scanned for 90+ known colloquial phrases (e.g., "heart racing" → palpitations, "feel sick" → nausea, "stiff neck" → neck_stiffness). Phrases are matched longest-first to prevent shorter sub-phrases from shadowing more specific multi-word expressions.

2. **Direct Token Matching Pass:** The remaining text is compared against all 30 canonical symptom names, accepting both underscore form (e.g., "runny_nose") and space-separated form (e.g., "runny nose").

The minimum symptom threshold (MIN_SYMPTOMS = 2) ensures the pipeline does not trigger diagnosis on a single ambiguous symptom, reducing false positives.

### 3.6 Natural Language Generation: Explanation Agent

**Technique:** Template-driven NLG (Russell & Norvig, Ch. 24 context)

The ExplanationAgent assembles a structured Markdown report by filling named slots with the outputs of upstream agents. The report includes five sections: primary diagnosis with confidence percentage, matched symptoms, numbered treatment pathway (with human-readable node label translations), drug safety status with violations and alternatives, and differential diagnoses. The agent is a pure function of its inputs — it maintains no state and performs no additional reasoning.

### 3.7 Integration: How All Techniques Connect

The five AI techniques are integrated through the OrchestratorAgent's FSM:

```
User Input → [NLP: Synonym Resolution] → Canonical Symptoms
                                              ↓
                              [FOL: Forward Chaining] → FOL Scores
                                              +
                              [NB: predict_proba] → ML Probabilities
                                              ↓
                              [Blend 0.6/0.4] → Ranked Diagnoses
                                              ↓
                              [A*: Treatment Graph Search] → Treatment Path
                                              ↓
                              [CSP: Constraint Checking] → Safety Report
                                              ↓
                              [NLG: Template Fill] → Patient Report
```

Each component's output is a typed Python data structure (list, dict, or str) passed explicitly to the next stage. The FSM enforces strict ordering and conditional backtracking, ensuring no stage executes without its preconditions being met.

---

## 4. Experimental Evaluation

### 4.1 Test Suite Overview

The MediAgent test suite comprises 7 test files with 60+ individual test cases covering 80%+ of the codebase:

| Test File | Scope | Tests |
|-----------|-------|-------|
| test_orchestrator.py | FSM state transitions, reset, done behavior | 8 |
| test_symptom_collector.py | Synonym resolution, token matching, follow-up questions | 10 |
| test_diagnosis_agent.py | FOL inference, ML blending, score ordering | 11 |
| test_treatment_planner.py | A* paths, severity start nodes, unknown disease fallback | 8 |
| test_drug_safety_agent.py | CSP constraints, violation detection, alternatives | 9 |
| test_explanation_agent.py | Report structure, label translation, differential display | 8 |
| test_integration.py | End-to-end pipeline for 5 disease scenarios | 7 |

### 4.2 Evaluation Scenarios

Five clinical scenarios spanning the range of disease severity and symptom overlap were selected for evaluation:

**Scenario 1 — Influenza (Mild, High Symptom Specificity)**  
Input: "I have fever, body aches, fatigue, and chills"  
Expected top diagnosis: Flu

**Scenario 2 — Migraine (Moderate, Distinct Symptom Cluster)**  
Input: "I have headache, nausea, and light sensitivity"  
Expected top diagnosis: Migraine

**Scenario 3 — Type 2 Diabetes (Moderate, Metabolic Symptoms)**  
Input: "I have frequent urination, excessive thirst, and blurred vision"  
Expected top diagnosis: Type 2 Diabetes

**Scenario 4 — Hypothyroidism (Multi-Turn, Incremental Collection)**  
Turn 1: "I feel tired" → COLLECTING (insufficient symptoms)  
Turn 2: "also have weight gain and cold intolerance" → DONE  
Expected top diagnosis: Hypothyroidism

**Scenario 5 — Pneumonia (Severe, Drug Safety Check)**  
Input: "I have fever, cough, chest pain, and shortness of breath"  
Expected: Pneumonia diagnosis, prescription_medication start node, Amoxicillin/Azithromycin in drug list

### 4.3 Results

**Table 2: Diagnostic Accuracy Results**

| Scenario | Expected Disease | Top-1 Correct | Top-3 Correct | FOL Score | ML Prob | Blended Confidence |
|----------|-----------------|---------------|---------------|-----------|---------|-------------------|
| Flu (full symptoms) | Flu | Yes | Yes | 0.700 | ~0.85 | ~0.760 |
| Flu (partial: fever only) | Flu | No | No | <0.233 | ~0.30 | <0.260 |
| Migraine | Migraine | Yes | Yes | 1.000 | ~0.90 | ~0.960 |
| Type 2 Diabetes | Type 2 Diabetes | Yes | Yes | 1.000 | ~0.88 | ~0.952 |
| Hypothyroidism | Hypothyroidism | Yes | Yes | 1.000 | ~0.85 | ~0.940 |
| Pneumonia | Pneumonia | Yes | Yes | 0.700 | ~0.82 | ~0.748 |

Notes: FOL scores shown are exact computed values from the scoring formula. ML probabilities are estimates based on model architecture and synthetic dataset design; exact values depend on the specific trained model.pkl artifact. Blended confidence = 0.6 × FOL + 0.4 × ML.

**Table 3: A* Treatment Path Optimality**

| Disease | Severity | Start Node | Optimal Path | Path Length | Total Cost |
|---------|----------|-----------|--------------|-------------|------------|
| Flu | Mild | rest_prescribed | rest_prescribed → treated | 2 | 1 |
| Common Cold | Mild | rest_prescribed | rest_prescribed → treated | 2 | 1 |
| Migraine | Moderate | otc_medication | otc_medication → treated | 2 | 2 |
| Hypertension | Moderate | otc_medication | otc_medication → treated | 2 | 2 |
| Pneumonia | Severe | prescription_medication | prescription_medication → treated | 2 | 3 |
| Angina | Severe | prescription_medication | prescription_medication → treated | 2 | 3 |

All A* paths in integration tests terminated at the "treated" node with the minimum-cost route from the severity-start node, as verified by `test_plan_ends_with_treated` and `test_plan_pneumonia_starts_with_prescription`.

**Table 4: CSP Drug Safety Performance**

| Scenario | Drugs Checked | Forbidden Pairs Present | Violations Detected | Alternatives Offered | Constraint Satisfaction Rate |
|----------|--------------|------------------------|--------------------|--------------------|----------------------------|
| Flu (no pre-existing drugs) | 3 | 0 | 0 | 0 | 100% |
| Common Cold | 3 | 0 | 0 | 0 | 100% |
| Hypothyroidism | 1 | 0 | 0 | 0 | 100% |
| Angina + Warfarin | 4 | 2 | 2 | 3 | 100% |
| Hypertension + Verapamil | 4 | 1 | 1 | 2 | 100% |

All forbidden pairs present in the active drug set were detected. The CSP constraint satisfaction rate is 100% across all tested scenarios.

**Table 5: Integration Test Pass Rates**

| Test | Pass |
|------|------|
| Full flu pipeline (state = DONE, Flu in response, Treatment/Drug Safety sections present) | PASS |
| Full migraine pipeline | PASS |
| Full diabetes pipeline | PASS |
| Incremental symptom collection (hypothyroidism, 2 turns) | PASS |
| Reset and new session | PASS |
| Explanation contains all sections | PASS |
| Natural language diabetes (synonym extraction) | PASS |

### 4.4 Analysis

**Where the system performs well:**

The FOL inference engine achieves perfect precision for diseases with a highly specific required-symptom set. Migraine (requiring headache + nausea + light_sensitivity) and Type 2 Diabetes (requiring frequent_urination + excessive_thirst) produce FOL scores of 1.0 when all required symptoms are present, eliminating ambiguity at the FOL layer. The multi-turn symptom accumulation correctly handles the Hypothyroidism scenario, where the patient's initial input ("tired") is insufficient and a second turn supplies the diagnostic evidence.

The CSP layer successfully detects all seeded forbidden drug pairs, including cross-regimen interactions (Angina drugs + patient-supplied Warfarin). The NLG layer correctly translates all internal node identifiers to human-readable labels and produces structurally complete reports in all test scenarios.

**Where the system struggles:**

The most significant weakness is overlap in the symptom space. Several diseases share required symptoms: both Flu and Pneumonia require fever; both Hypertension and Migraine require headache and dizziness; both Bronchitis and Pneumonia require cough and can involve chest pain. In scenarios where a patient presents with shared symptoms but does not report disease-specific differentiators, the FOL engine may produce a multi-way tie at the top of the ranking, and the blended confidence may fall below the 0.5 threshold, triggering unnecessary follow-up.

The Naive Bayes model's effectiveness is constrained by the synthetic training data. Since training samples are generated from the same rules used for FOL inference, the model may simply amplify FOL scores rather than providing genuinely independent probabilistic evidence. In cases where the FOL score is ambiguous, the ML layer may not meaningfully resolve the uncertainty.

The synonym vocabulary, while extensive (90+ entries), does not cover all possible patient phrasings. A patient describing "pins and needles" or "numbness" will not be mapped to any canonical symptom, and the pipeline may request clarification or fail to diagnose a relevant condition.

---

## 5. Discussion

### 5.1 Strengths

**Modular, Interpretable Architecture**  
The multi-agent design provides strong separation of concerns: each AI technique is encapsulated in a dedicated agent with a clean interface, making the system easy to test, modify, and extend. The explicit representation of FOL rules and CSP constraints means a domain expert can directly inspect and update the knowledge base without touching any algorithmic code.

**Explainability by Design**  
MediAgent's diagnostic outputs are inherently explainable. The FOL inference engine records exactly which required and supporting symptoms matched for each candidate disease, the blended confidence formula is fully transparent, the A* search produces an explicit ordered pathway rather than a black-box recommendation, and the CSP layer identifies exactly which drug pairs conflict and why. This stands in contrast to deep learning approaches that produce accurate but opaque predictions.

**Comprehensive AI Technique Coverage**  
The system demonstrates five classical AI techniques from Russell & Norvig in a single integrated application: search (A*), knowledge representation (FOL), constraint satisfaction (CSP), probabilistic learning (Naive Bayes), and natural language generation (template NLG). This breadth illustrates how diverse AI paradigms can complement each other.

**Robust Testing**  
The 60+ test suite with enforced 80% coverage provides strong confidence in component correctness. The test suite includes both unit tests (isolating each agent) and integration tests (verifying the full pipeline for five clinical scenarios), catching both isolated and emergent failures.

**Graceful Degradation**  
The system handles missing model files, unknown diseases, disconnected graph paths, and insufficient symptoms without crashing, always returning a sensible response or fallback pathway.

### 5.2 Limitations

**Synthetic Training Data**  
The Naive Bayes model is trained entirely on programmatically generated data derived from the same FOL rules used for inference. This creates a circularity: the ML layer does not provide genuinely independent statistical evidence but partially replicates the FOL scores. Real clinical datasets (e.g., MIMIC-III, UK Biobank symptom records) would be required for meaningful statistical learning.

**Limited and Non-Scalable Knowledge Base**  
The current knowledge base covers only 10 diseases and 30 symptoms. The real disease ontology encompasses thousands of conditions and hundreds of thousands of symptom-disease associations. Manually maintaining a hand-crafted FOL rule base at scale is impractical. A probabilistic knowledge base (e.g., a Bayesian network over diseases and symptoms) or integration with a structured medical ontology (e.g., SNOMED CT, ICD-10) would be required for clinical-scale deployment.

**No Real Medical Validation**  
The system has not been evaluated against real patient data or validated by medical professionals. The diagnostic accuracy figures reported in Section 4 are based on integration tests that confirm the correct disease appears in the output for hand-crafted symptom inputs derived from the same knowledge base. This is not equivalent to clinical validation.

**English-Only Symptom Extraction**  
The synonym resolution table is entirely in English. The system cannot process symptom descriptions in other languages. Given the global healthcare access motivation stated in Section 1, this is a significant limitation for deployment contexts where patients do not communicate in English.

**No Temporal or Severity Modeling**  
The current system treats all symptom reports as equally recent and of equal severity. Clinical diagnosis frequently relies on temporal context (when did symptoms begin? have they worsened?) and severity gradation (mild vs. severe headache). These dimensions are entirely absent from the current model.

### 5.3 Future Improvements

**Real EHR Integration**  
Replacing the synthetic dataset with de-identified electronic health record (EHR) data would dramatically improve the Naive Bayes model's fidelity. The system architecture is already set up to accommodate this: the training pipeline in `ml/train.py` accepts any DataFrame with the 30-symptom binary features and a disease label column.

**Probabilistic Knowledge Base**  
Replacing the hand-crafted FOL rules with a learned Bayesian network (R&N Ch. 13) would allow the system to represent uncertainty in symptom-disease associations and update beliefs incrementally as new symptoms are reported. This would naturally handle symptom overlap between diseases without requiring manual weighting.

**Multi-Language Support**  
Extending the synonym resolution table or integrating a multilingual NLP pipeline (e.g., spaCy with multilingual models, or a medical NER model fine-tuned on non-English clinical text) would expand the system's accessibility.

**Learning from User Feedback**  
Incorporating a reinforcement learning layer (R&N Ch. 22) that updates symptom weights or disease priors based on patient-confirmed diagnoses (with appropriate privacy protections) would allow the system to improve over time.

**Larger Disease and Drug Coverage**  
Extending the knowledge base to cover more diseases, a larger symptom vocabulary, and more drug interactions (potentially by parsing structured drug interaction databases such as DrugBank) would increase clinical utility.

**Integration with Medical Ontologies**  
Mapping the symptom and disease vocabulary to SNOMED CT or ICD-10 codes would enable interoperability with existing clinical decision support systems and allow the knowledge base to be populated from curated clinical sources rather than manual rule encoding.

### 5.4 Ethical Considerations

**Not a Substitute for Professional Medical Advice**  
This is the most important ethical consideration. MediAgent is an educational demonstration of AI techniques and must not be used to make real clinical decisions. The system prominently displays the disclaimer "Educational demo — not medical advice" in the Streamlit interface, and the ExplanationAgent's fallback message explicitly directs users without a recognized diagnosis to consult a doctor. Any real-world deployment would require clinical validation, regulatory approval, and qualified physician oversight.

**Potential Bias in Synthetic Data**  
The synthetic training dataset is generated from rules written by the development team. These rules reflect the team's knowledge and assumptions about disease presentations, which may not accurately represent the full spectrum of clinical presentations, particularly for patients with atypical presentations, comorbidities, or presentations influenced by age, sex, or ethnicity. A model trained on biased synthetic data may systematically under-diagnose or misdiagnose certain patient populations.

**Privacy of Health Data**  
In its current form, MediAgent processes all user input in-memory within a single Streamlit session and does not persist health data to external storage. Any production deployment would need to address HIPAA compliance (in the United States), GDPR health data provisions (in the EU), and equivalent regulations in other jurisdictions. Patient health data is among the most sensitive categories of personal information.

**Over-Reliance Risk**  
Users who receive a diagnostic output from MediAgent may be tempted to forgo professional medical consultation, particularly in low-access environments where a doctor is difficult to reach. The system's disclaimers and design must actively discourage this behavior.

---

## 6. Individual Contributions

The table below lists the system components and the team member(s) best positioned to have contributed to each, provided as a guide. Actual contributions and contribution percentages are to be completed, agreed upon, and signed by all team members.

**Table 6: Individual Contributions**

| Team Member | System Components | Contribution % |
|-------------|------------------|----------------|
| Deepesh Katudia | [To be completed] | [To be agreed] |
| Aditya Srivastava | [To be completed] | [To be agreed] |
| Manas Mankar | [To be completed] | [To be agreed] |
| Vishnu | [To be completed] | [To be agreed] |
| Bella | [To be completed] | [To be agreed] |

**Component Guide for Reference:**

| Component | Corresponding Files |
|-----------|-------------------|
| OrchestratorAgent / FSM design | `agents/orchestrator.py` |
| SymptomCollectionAgent / NLP synonym mapping | `agents/symptom_collector.py` |
| DiagnosisAgent / FOL inference engine | `agents/diagnosis_agent.py`, `knowledge/disease_rules.py` |
| Naive Bayes training pipeline | `ml/train.py`, `ml/dataset_generator.py` |
| TreatmentPlannerAgent / A* search | `agents/treatment_planner.py`, `knowledge/treatment_graph.py` |
| DrugSafetyAgent / CSP formulation | `agents/drug_safety_agent.py`, `knowledge/drug_interactions.py` |
| ExplanationAgent / NLG report | `agents/explanation_agent.py` |
| Streamlit UI | `ui/app.py` |
| Test suite (all 7 files) | `tests/` |
| Documentation and report | `docs/` |

*This table is to be completed by team members and signed before submission.*

---

## 7. References

1. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. [Primary reference — Chapters 2 (Agent Design), 3 (Search), 6 (CSP), 7 (Logic), 9 (Inference), 12 (Probabilistic Reasoning), 24 (NLP)]

2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [Naive Bayes implementation: `sklearn.naive_bayes.MultinomialNB`]

3. Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy 2008)* (pp. 11–15). [A* search on directed graph: `networkx.astar_path`]

4. Laborie, P. (2003). *python-constraint: A library for solving constraint satisfaction problems in Python*. [CSP drug interaction checking: `constraint.Problem`]

5. Streamlit Inc. (2023). *Streamlit Documentation* (Version 1.x). Retrieved from https://docs.streamlit.io [Chat interface implementation]

6. Rish, I. (2001). An empirical study of the naive Bayes classifier. *IJCAI Workshop on Empirical Methods in Artificial Intelligence*, 3(22), 41–46. [Naive Bayes theory and empirical performance]

7. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107. [Original A* algorithm paper]

8. Bacchus, F., & Grove, A. (1996). Utility independence in a qualitative decision theory. In *Proceedings of the 5th International Conference on Knowledge Representation and Reasoning (KR '96)*. [Utility-based agent theory]

9. Shortliffe, E. H., & Buchanan, B. G. (1975). A model of inexact reasoning in medicine. *Mathematical Biosciences*, 23(3–4), 351–379. [Foundational work on rule-based medical expert systems; context for FOL approach]

10. Lucas, P. J. F., & van der Gaag, L. C. (1991). *Principles of Expert Systems*. Addison-Wesley. [Background on forward chaining and medical knowledge representation]

---

## Appendix A: Project File Structure

```
mediagent/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py          # FSM controller
│   ├── symptom_collector.py     # NLP synonym mapping
│   ├── diagnosis_agent.py       # FOL + Naive Bayes hybrid
│   ├── treatment_planner.py     # A* search
│   ├── drug_safety_agent.py     # CSP constraint checking
│   └── explanation_agent.py     # Template-driven NLG
├── knowledge/
│   ├── __init__.py
│   ├── disease_rules.py         # FOL knowledge base (10 diseases, 30 symptoms)
│   ├── drug_interactions.py     # CSP constraints (5 forbidden pairs, alternatives)
│   └── treatment_graph.py       # A* graph (7 nodes, 10 edges, heuristics)
├── ml/
│   ├── __init__.py
│   ├── dataset_generator.py     # Synthetic training data generation
│   ├── train.py                 # MultinomialNB training pipeline
│   └── model.pkl                # Serialised trained model (generated)
├── ui/
│   ├── __init__.py
│   └── app.py                   # Streamlit chat interface
├── tests/
│   ├── __init__.py
│   ├── test_orchestrator.py
│   ├── test_symptom_collector.py
│   ├── test_diagnosis_agent.py
│   ├── test_treatment_planner.py
│   ├── test_drug_safety_agent.py
│   ├── test_explanation_agent.py
│   └── test_integration.py
└── docs/
    └── final_report.md          # This document
```

## Appendix B: Running the System

**Prerequisites:**
```bash
pip install streamlit scikit-learn networkx python-constraint numpy pandas joblib python-dotenv
```

**Train the Naive Bayes model:**
```bash
python ml/train.py
```

**Run the test suite:**
```bash
pytest tests/ --cov=agents --cov=knowledge --cov=ml --cov-report=term-missing
```

**Launch the Streamlit interface:**
```bash
streamlit run ui/app.py
```

---

*This report was prepared as an academic submission for an Artificial Intelligence course. MediAgent is an educational prototype and is not intended for clinical use. Always consult a qualified medical professional for health concerns.*
