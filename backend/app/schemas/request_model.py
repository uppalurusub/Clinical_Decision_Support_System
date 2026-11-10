from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ClinicalDecisionRequest(BaseModel):

    patient_id: str

    symptoms: Dict[str, Any]

    history: Optional[Dict[str, Any]] = None

    vitals: Optional[Dict[str, Any]] = None

    medications: Optional[List[Dict[str, Any]]] = None

    allergies: Optional[List[str]] = None

    labs: Optional[List[Dict[str, Any]]] = None

    imaging: Optional[List[Dict[str, Any]]] = None