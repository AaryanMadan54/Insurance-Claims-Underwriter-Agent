import requests

def lookup_clinical_codes(description: str):
    """Calls the NIH API to find official ICD-10 diagnosis codes."""
    base_url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {"sf": "code,name", "terms": description, "maxList": 3}
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        return [{"code": r[0], "desc": r[1]} for r in data[3]] if data[0] > 0 else "No code found."
    except: return "API Offline"

def validate_cpt_2026(cpt_code: str):
    """Checks 2026 CPT codes, including new AI-augmented procedures."""
    rules = {
        "99213": "Office visit (Covered)",
        "0902T": "AI-Augmented EKG (Requires 2026 AI-Surcharge Approval)", # Modern 2026 code
        "92002": "Eye Exam (Requires Specialist Auth)"
    }
    return rules.get(cpt_code, "Manual Review Required")


def check_medical_necessity(cpt_code: str, icd10_code: str):
    """
    Ensures the Procedure (CPT) is justified by the Diagnosis (ICD-10).
    This is the "Secret Sauce" that prevents insurance denials.
    """
    # Mapping valid pairs (In 2026, you'd pull this from an NCCI database)
    valid_pairings = {
        "99213": ["J02.9", "R05", "Z00.00"], # Office visit is valid for sore throat, cough, checkup
        "70551": ["G43.909", "R51.9"],       # MRI Brain is valid for Migraine, Headache
        "29881": ["S83.242A"],               # Knee Arthroscopy valid for Meniscus Tear
    }
    
    # Clean the codes (strip decimals for easier matching)
    clean_icd = icd10_code.replace(".", "")
    
    if cpt_code in valid_pairings:
        # Check if the diagnosis provided is in the approved list
        if any(clean_icd.startswith(approved.replace(".", "")) for approved in valid_pairings[cpt_code]):
            return {"status": "MATCH", "message": "Procedure justified by diagnosis."}
        return {"status": "MISMATCH", "message": f"Procedure {cpt_code} not typically indicated for diagnosis {icd10_code}."}
    
    return {"status": "UNKNOWN", "message": "New/Complex pairing: Requires Clinical Review."}