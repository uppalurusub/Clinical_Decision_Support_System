from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ClinicalDecisionResponse(BaseModel):

    patient_id: str

    clinical_summary: str

    differential_diagnoses: List[Dict[str, Any]]

    missing_information: List[Dict[str, Any]]

    suggested_tests: List[Dict[str, Any]]

    safety_flags: List[Dict[str, Any]]

    evidence: List[Dict[str, Any]]

    requires_human_review: bool

    workflow_id: str