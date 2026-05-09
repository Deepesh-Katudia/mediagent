# MediAgent — Data Flow Reference

This document describes the complete data flow through the MediAgent system: from raw patient input to the final diagnostic report. It covers the data structures exchanged between agents, a detailed worked example, and the handling of error and edge cases.

---

## 1. Step-by-Step Data Flow

The following table summarises every transformation applied to the patient's input as it moves through the six-agent pipeline.

| Step | Agent | Input | Output | Notes |
|---|---|---|---|---|
| 1 | **UI** (`ui/app.py`) | User types free-text | `prompt: str` passed to `OrchestratorAgent.process()` | Streamlit `st.chat_input` captures the raw string |
| 2 | **OrchestratorAgent** | `prompt: str` | Routes to appropriate specialist agent based on current FSM state | Single entry point; accumulates `symptoms`, `diagnosis`, `treatment`, `safety` across turns |
| 3 | **SymptomCollectionAgent** | `prompt: str` | `symptoms: list[str]` — sorted canonical tokens | Two-pass NLP: synonym map → direct token match |
| 4 | **OrchestratorAgent** | `symptoms: list[str]` | Either a follow-up question `str` (if `len(symptoms) < 2`) or advances to DIAGNOSING | Deduplicates and sorts the cumulative symptom list |
| 5 | **DiagnosisAgent** | `symptoms: list[str]` | `diagnosis: list[dict]` — up to 3 ranked candidates | FOL inference + Naive Bayes blending |
| 6 | **OrchestratorAgent** | `diagnosis: list[dict]` | Advances to PLANNING or backtracks to COLLECTING | Confidence gate at 0.50 |
| 7 | **TreatmentPlannerAgent** | `disease: str` (top diagnosis) | `treatment: list[str]` — ordered node names | A* search on treatment graph |
| 8 | **DrugSafetyAgent** | `disease: str` (top diagnosis) | `safety: dict` — safe flag, drugs, violations, alternatives | CSP drug interaction check |
| 9 | **ExplanationAgent** | `symptoms`, `diagnosis`, `treatment`, `safety` | `report: str` — Markdown-formatted narrative | Pure formatter; no additional reasoning |
| 10 | **UI** | `report: str` | Rendered in Streamlit chat bubble | `st.markdown()` renders Markdown |

---

## 2. Data Structures Between Agents

### 2.1 SymptomCollectionAgent Output

`extract_symptoms(text: str) → list[str]`

```python
# Example return value
["body_aches", "fatigue", "fever", "headache"]
```

- Each element is a canonical symptom token from the `SYMPTOMS` vocabulary (30 tokens).
- The list is sorted alphabetically.
- The OrchestratorAgent merges each turn's output with the accumulated list using `sorted(set(accumulated + new))`.

`get_followup_question(symptoms: list[str]) → str`

```python
# No symptoms yet
"Please describe your symptoms. What are you experiencing?"

# Some symptoms collected
"I've noted: body_aches, fever. Are there any other symptoms you're experiencing?"
```

### 2.2 DiagnosisAgent Output

`diagnose(symptoms: list[str]) → list[dict]`

```python
[
    {
        "disease":            "Flu",
        "fol_score":          0.775,
        "ml_prob":            0.821,
        "confidence":         0.793,          # 0.6*fol_score + 0.4*ml_prob
        "matched_required":   ["body_aches", "fatigue", "fever"],
        "matched_supporting": ["headache"],
    },
    {
        "disease":            "Pneumonia",
        "fol_score":          0.233,
        "ml_prob":            0.061,
        "confidence":         0.164,
        "matched_required":   ["fever"],
        "matched_supporting": [],
    },
]
```

- Sorted descending by `confidence`.
- Contains at most 3 candidates (top-3 from FOL, re-sorted after blending).
- `confidence` is the gating value compared against `CONFIDENCE_THRESHOLD = 0.50`.

### 2.3 TreatmentPlannerAgent Output

`plan_treatment(disease: str) → list[str]`

```python
# Flu (mild severity) — start node: "rest_prescribed"
["rest_prescribed", "treated"]

# Pneumonia (severe severity) — start node: "prescription_medication"
["prescription_medication", "treated"]

# Hypertension (moderate severity) — start node: "otc_medication"
["otc_medication", "prescription_medication", "treated"]
```

- Ordered list of treatment node names from start to goal.
- The ExplanationAgent maps these to human-readable labels via `step_labels`.
- Fallback path on `NetworkXNoPath`: `[start, "specialist_referral", "treated"]`.

### 2.4 DrugSafetyAgent Output

`check_safety(disease: str, existing_drugs: list[str] = []) → dict`

```python
# No interactions (e.g., Flu with default drugs)
{
    "safe":         True,
    "drugs":        ["Oseltamivir", "Ibuprofen", "Acetaminophen"],
    "violations":   [],
    "alternatives": {},
}

# Interaction detected (e.g., Angina patient also taking Warfarin)
{
    "safe":         False,
    "drugs":        ["Aspirin", "Beta_Blockers", "Nitroglycerin", "Warfarin"],
    "violations":   ["Aspirin + Warfarin interaction detected"],
    "alternatives": {"Aspirin": "Acetaminophen", "Warfarin": "Dabigatran"},
}
```

### 2.5 ExplanationAgent Output

`explain(symptoms, diagnosis, treatment, safety) → str`

The output is a multi-line Markdown string. Example:

```
**Diagnosis:** Flu (79% confidence)
**Matched symptoms:** body_aches, fatigue, fever, headache

**Treatment Plan:**
  1. Rest and hydration
  2. Recovery expected

**Drug Safety:** Safe to use — Oseltamivir, Ibuprofen, Acetaminophen

**Other possibilities considered:**
  - Pneumonia (16%)
```

---

## 3. Example Trace: "I have fever, headache, and body aches"

This section traces a complete single-turn consultation through all six agents.

### Turn 1 — Patient Input

```
User: "I have fever, headache, and body aches"
```

**OrchestratorAgent** receives the input. Current state: `COLLECTING`.

---

### Step 1 — SymptomCollectionAgent

`extract_symptoms("I have fever, headache, and body aches")`

**Synonym pass** (longest phrase first):
- `"body aches"` → matches `SYMPTOM_SYNONYMS["body aches"]` → canonical: `"body_aches"`
- `"fever"` → matched via direct token scan (in `SYMPTOMS`)
- `"headache"` → matched via direct token scan (in `SYMPTOMS`)

**Result:**
```python
new_symptoms = ["body_aches", "fever", "headache"]
```

**Orchestrator accumulates:**
```python
self.symptoms = sorted(set([] + ["body_aches", "fever", "headache"]))
             = ["body_aches", "fever", "headache"]
```

`has_enough_symptoms(["body_aches", "fever", "headache"])` → `3 >= 2` → `True`

State advances to `DIAGNOSING`.

---

### Step 2 — DiagnosisAgent (FOL Inference)

`run_fol_inference(["body_aches", "fever", "headache"])`

**Flu evaluation:**
- required = {fever, body_aches, fatigue}; matches = {fever, body_aches} → 2/3
- supporting = {chills, headache, cough, sore_throat}; matches = {headache} → 1/4
- fol_score = (2/3 × 0.7) + (1/4 × 0.3) = 0.467 + 0.075 = **0.542**

**Pneumonia evaluation:**
- required = {fever, cough, chest_pain}; matches = {fever} → 1/3
- supporting = {shortness_of_breath, fatigue, sweating, chills}; matches = {} → 0/4
- fol_score = (1/3 × 0.7) + (0 × 0.3) = **0.233**

**Hypertension evaluation:**
- required = {headache, dizziness}; matches = {headache} only → 1/2
- But required_matches = 1 > 0, so eligible
- fol_score = (1/2 × 0.7) + (0/3 × 0.3) = **0.350**

**Tension Headache evaluation:**
- required = {headache}; matches = {headache} → 1/1
- supporting = {neck_stiffness, fatigue, dizziness}; matches = {} → 0/3
- fol_score = (1/1 × 0.7) + (0/3 × 0.3) = **0.700**

**Top 3 by fol_score:** Tension Headache (0.700), Flu (0.542), Hypertension (0.350)

---

### Step 3 — DiagnosisAgent (Naive Bayes)

`run_ml_inference(["body_aches", "fever", "headache"])`

Binary feature vector (30 elements, showing non-zero):
```
fever=1, body_aches=1, headache=1, all others=0
```

Classifier returns posterior probabilities. Example output (approximate):
```python
{
    "Flu":              0.521,
    "Tension Headache": 0.183,
    "Hypertension":     0.091,
    # ... other diseases with low probability
}
```

---

### Step 4 — DiagnosisAgent (Blending)

```
Flu:              confidence = 0.6 × 0.542 + 0.4 × 0.521 = 0.325 + 0.208 = 0.533
Tension Headache: confidence = 0.6 × 0.700 + 0.4 × 0.183 = 0.420 + 0.073 = 0.493
Hypertension:     confidence = 0.6 × 0.350 + 0.4 × 0.091 = 0.210 + 0.036 = 0.246
```

Final ranked diagnosis list:
```python
[
    {"disease": "Flu",              "confidence": 0.533, "fol_score": 0.542, "ml_prob": 0.521,
     "matched_required": ["body_aches", "fever"], "matched_supporting": ["headache"]},
    {"disease": "Tension Headache", "confidence": 0.493, "fol_score": 0.700, "ml_prob": 0.183,
     "matched_required": ["headache"], "matched_supporting": []},
    {"disease": "Hypertension",     "confidence": 0.246, "fol_score": 0.350, "ml_prob": 0.091,
     "matched_required": ["headache"], "matched_supporting": []},
]
```

`diagnosis[0].confidence = 0.533 >= 0.50` → passes gate → state advances to `PLANNING`.

---

### Step 5 — TreatmentPlannerAgent

`plan_treatment("Flu")`

- `DISEASE_SEVERITY["Flu"]` = `"mild"`
- `SEVERITY_START["mild"]` = `"rest_prescribed"`
- A* from `"rest_prescribed"` to `"treated"`:
  - Direct edge cost: 1
  - Alternative via `otc_medication`: cost 1 + 2 = 3
  - A* selects minimum: `["rest_prescribed", "treated"]`

```python
treatment = ["rest_prescribed", "treated"]
```

State advances to `CHECKING`.

---

### Step 6 — DrugSafetyAgent

`check_safety("Flu")`

- `DISEASE_DRUGS["Flu"]` = `["Oseltamivir", "Ibuprofen", "Acetaminophen"]`
- No existing drugs provided.
- Check `FORBIDDEN_PAIRS`:
  - `(Aspirin, Warfarin)` — Aspirin not in drug list → skip
  - `(Ibuprofen, Warfarin)` — Warfarin not in drug list → skip
  - All other pairs — neither drug in list → skip
- No violations.

```python
safety = {
    "safe":         True,
    "drugs":        ["Oseltamivir", "Ibuprofen", "Acetaminophen"],
    "violations":   [],
    "alternatives": {},
}
```

State advances to `EXPLAINING`.

---

### Step 7 — ExplanationAgent

`explain(["body_aches", "fever", "headache"], diagnosis, ["rest_prescribed", "treated"], safety)`

Assembled report:
```
**Diagnosis:** Flu (53% confidence)
**Matched symptoms:** body_aches, fever, headache

**Treatment Plan:**
  1. Rest and hydration
  2. Recovery expected

**Drug Safety:** Safe to use — Oseltamivir, Ibuprofen, Acetaminophen

**Other possibilities considered:**
  - Tension Headache (49%)
  - Hypertension (25%)
```

State advances to `DONE`. Report is returned to the UI and rendered in the chat bubble.

---

## 4. Error and Edge Cases

### 4.1 No Symptoms Detected

**Condition:** The patient's input contains no recognisable symptom tokens or synonyms.

**Trigger:** `extract_symptoms("Hello, how are you?")` returns `[]`.

**Handling:**
```
symptoms = []
has_enough_symptoms([]) → False
get_followup_question([]) → "Please describe your symptoms. What are you experiencing?"
```

The FSM stays in `COLLECTING`. The system returns the generic prompt and waits for the next user turn. The patient may submit multiple turns; symptoms accumulate across all of them.

---

### 4.2 Insufficient Symptoms After Several Turns

**Condition:** Only one symptom has been extracted over multiple turns and the patient has not provided more.

**Trigger:** `has_enough_symptoms(["fever"]) → False` (1 < MIN_SYMPTOMS = 2).

**Handling:**
```
get_followup_question(["fever"])
→ "I've noted: fever. Are there any other symptoms you're experiencing?"
```

The FSM remains in `COLLECTING` indefinitely until at least 2 unique canonical symptoms are accumulated. There is no turn limit; the session does not time out automatically.

---

### 4.3 Low-Confidence Diagnosis (FSM Backtrack)

**Condition:** FOL inference finds candidates but all blended confidence scores fall below the 0.50 threshold.

**Trigger:** `diagnosis[0]["confidence"] < CONFIDENCE_THRESHOLD (0.50)`.

**Handling in OrchestratorAgent:**
```python
self.state = State.COLLECTING   # backtrack
return (
    f"I have your symptoms ({', '.join(self.symptoms)}) but need more detail. "
    "Can you describe any additional symptoms?"
)
```

The system reverts to `COLLECTING`. Crucially, the accumulated `self.symptoms` list is **not cleared** — new symptoms provided in the next turn are merged with existing ones. This allows subsequent turns to provide the additional evidence needed to push confidence above the threshold.

**Example:** Patient reports only "headache". Multiple diseases match, but no single disease exceeds 0.50 confidence. The system requests more symptoms. Patient adds "nausea, light sensitivity" → Migraine now has high FOL coverage and confidence exceeds 0.50.

---

### 4.4 Disease Not Found in Treatment Graph

**Condition:** The `plan_treatment` call is made for a disease whose severity is not listed in `DISEASE_SEVERITY`.

**Handling:**
```python
severity = DISEASE_SEVERITY.get(disease, "moderate")   # default: moderate
start = SEVERITY_START.get(severity, self.DEFAULT_START)  # default: "otc_medication"
```

Unknown diseases default to `"moderate"` severity and start at `"otc_medication"`. If A* still fails to find a path (disconnected graph), the fallback is:
```python
return [start, "specialist_referral", "treated"]
```

This guarantees that `TreatmentPlannerAgent.plan_treatment()` always returns a non-empty list.

---

### 4.5 Drug Interactions Found

**Condition:** The top diagnosis includes drugs that appear in `FORBIDDEN_PAIRS`, or the patient is already taking a drug that interacts with a prescribed one.

**Example:** A Tension Headache patient is also taking Warfarin.

- `DISEASE_DRUGS["Tension Headache"]` = `["Ibuprofen", "Acetaminophen", "Aspirin"]`
- Forbidden pair `(Ibuprofen, Warfarin)` and `(Aspirin, Warfarin)` both fire.

**Safety report:**
```python
{
    "safe":       False,
    "drugs":      ["Ibuprofen", "Acetaminophen", "Aspirin", "Warfarin"],
    "violations": [
        "Ibuprofen + Warfarin interaction detected",
        "Aspirin + Warfarin interaction detected"
    ],
    "alternatives": {
        "Ibuprofen": "Acetaminophen",
        "Aspirin":   "Acetaminophen",
        "Warfarin":  "Dabigatran",
    },
}
```

**ExplanationAgent output:**
```
**Drug Safety:** Warning — interaction detected!
  - Ibuprofen + Warfarin interaction detected
  - Aspirin + Warfarin interaction detected
  Alternatives: Ibuprofen → Acetaminophen, Aspirin → Acetaminophen, Warfarin → Dabigatran
```

The pipeline does **not** stop or raise an error on interaction detection. The safety report is passed through to the ExplanationAgent, which includes the warning prominently in the final report. The user is expected to consult a clinician before acting on any recommendation.

---

### 4.6 Naive Bayes Model Not Found

**Condition:** `ml/model.pkl` does not exist (e.g., `ml/train.py` has not been run).

**Handling in DiagnosisAgent:**
```python
def _load_model(self):
    if os.path.exists(MODEL_PATH):
        self._model = joblib.load(MODEL_PATH)
    # else: self._model remains None

def run_ml_inference(self, symptoms):
    if self._model is None:
        return {}   # empty dict — no ML contribution
```

When `run_ml_inference` returns `{}`, the blending step defaults `ml_prob = 0.0` for all candidates:
```
confidence = 0.6 × fol_score + 0.4 × 0.0 = 0.6 × fol_score
```

The system operates in **pure FOL mode** with no error or exception. The confidence threshold (0.50) still applies; with FOL-only scoring, the system requires `fol_score >= 0.833` to pass the gate (since `0.6 × 0.833 ≈ 0.50`), which means higher required-symptom coverage is needed before the pipeline advances.

---

### 4.7 Empty Diagnosis List

**Condition:** The patient's symptoms do not match the required symptom set of any disease in `DISEASE_RULES`.

**Trigger:** `run_fol_inference` returns `[]` because every disease failed the `required_matches > 0` hard gate.

**Handling in OrchestratorAgent:**
```python
if not self.diagnosis or self.diagnosis[0]["confidence"] < CONFIDENCE_THRESHOLD:
    self.state = State.COLLECTING
    return "I have your symptoms (...) but need more detail."
```

**Handling in ExplanationAgent (called only if diagnosis is non-empty, but as a safety net):**
```python
if not diagnosis:
    return "I was unable to identify a condition from the symptoms provided. Please consult a doctor."
```

The system never crashes on an empty diagnosis list. It reverts to requesting more symptom information.
