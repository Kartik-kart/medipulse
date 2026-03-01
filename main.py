from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import json
import uuid

app = FastAPI(title="MediPulse API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory patient storage (resets on server restart)
PATIENTS = {
    "demo-001": {
        "id": "demo-001",
        "name": "Alex Rivera",
        "age": 47,
        "blood_type": "A+",
        "surgeries": [
            {"name": "Appendectomy", "year": 2015},
            {"name": "Right Knee Arthroscopy", "year": 2020}
        ],
        "conditions": ["Type 2 Diabetes", "Hypertension", "Mild Asthma"],
        "allergies": [
            {"substance": "Penicillin", "reaction": "Anaphylaxis"},
            {"substance": "Ibuprofen", "reaction": "Hives"},
            {"substance": "Shellfish", "reaction": "Swelling"}
        ],
        "medications": [
            {"name": "Metformin", "dose": "500mg", "frequency": "Twice daily"},
            {"name": "Lisinopril", "dose": "10mg", "frequency": "Once daily"},
            {"name": "Albuterol Inhaler", "dose": "90mcg", "frequency": "As needed"}
        ]
    }
}

class Surgery(BaseModel):
    name: str
    year: int

class Allergy(BaseModel):
    substance: str
    reaction: str

class Medication(BaseModel):
    name: str
    dose: str
    frequency: str

class Patient(BaseModel):
    name: str
    age: int
    blood_type: str
    surgeries: List[Surgery] = []
    conditions: List[str] = []
    allergies: List[Allergy] = []
    medications: List[Medication] = []

class SymptomRequest(BaseModel):
    symptom: str
    patient_id: str

@app.get("/")
def home():
    return {"status": "MediPulse API is running", "version": "2.0.0"}

@app.get("/patients")
def get_all_patients():
    return list(PATIENTS.values())

@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PATIENTS[patient_id]

@app.post("/patients")
def create_patient(patient: Patient):
    patient_id = str(uuid.uuid4())[:8]
    PATIENTS[patient_id] = {"id": patient_id, **patient.dict()}
    return PATIENTS[patient_id]

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    if patient_id == "demo-001":
        raise HTTPException(status_code=400, detail="Cannot delete demo patient")
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")
    del PATIENTS[patient_id]
    return {"message": "Patient deleted"}

@app.post("/analyze")
async def analyze_symptom(request: SymptomRequest):
    if request.patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = PATIENTS[request.patient_id]
    symptom = request.symptom.strip()
    if not symptom:
        raise HTTPException(status_code=400, detail="Symptom cannot be empty")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    patient_summary = f"""
Patient: {patient['name']}, Age {patient['age']}, Blood Type {patient['blood_type']}
Conditions: {', '.join(patient['conditions']) or 'None'}
Allergies: {', '.join([f"{a['substance']} ({a['reaction']})" for a in patient['allergies']]) or 'None'}
Medications: {', '.join([f"{m['name']} {m['dose']}" for m in patient['medications']]) or 'None'}
Surgeries: {', '.join([f"{s['name']} ({s['year']})" for s in patient['surgeries']]) or 'None'}
"""

    prompt = f"""You are an emergency medical AI assistant. A patient presents with: "{symptom}"

Patient Medical History:
{patient_summary}

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{
  "possible_causes": ["cause 1", "cause 2", "cause 3"],
  "risk_level": "Low or Medium or High",
  "responder_checks": ["check 1", "check 2", "check 3"],
  "red_flags": ["flag 1", "flag 2"],
  "summary": "One paragraph summary for emergency responders"
}}"""

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5500",
                "X-Title": "MediPulse"
            },
            json={
                "model": "openrouter/auto",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AI API error: {response.text}")

    content = response.json()["choices"][0]["message"]["content"]

    try:
        clean = content.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            raise HTTPException(status_code=500, detail="Failed to parse AI response")

    risk_colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
    risk_level = parsed.get("risk_level", "Medium")

    return {
        "symptom": symptom,
        "patient_name": patient["name"],
        "possible_causes": parsed.get("possible_causes", []),
        "risk_level": risk_level,
        "risk_color": risk_colors.get(risk_level, "#f59e0b"),
        "responder_checks": parsed.get("responder_checks", []),
        "red_flags": parsed.get("red_flags", []),
        "summary": parsed.get("summary", "")
    }