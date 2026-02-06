import streamlit as st
import os
import easyocr
import json
import uuid
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.types import Command

# --- Import your updated workflow ---
from updated_agent_workflow import app, EnterpriseState

# --- Import RAG Engine (NEW) ---
from rag_engine import add_policy_to_system

# --- CONFIGURATION ---
TEMP_FILE_PATH = "temp_uploaded_bill.jpg"
st.set_page_config(page_title="2026 Claims Adjuster OS", layout="wide")
LLM_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF" 
OCR_READER = easyocr.Reader(['en'])

# --- SCHEMA DEFINITION ---
class ClaimLineItem(BaseModel):
    description: str
    amount: float

class StructuredClaimData(BaseModel):
    invoice_number: str
    patient_name: str
    total_claimed_amount: float
    line_items: List[ClaimLineItem]

# --- OCR & LLM EXTRACTION UTILITIES ---
@st.cache_resource
def extract_text_from_bill(image_path: str) -> str:
    results = OCR_READER.readtext(image_path, detail=0)
    ocr_text = " ".join(results)
    
    # DEBUG: Show what OCR read
    print("\n" + "="*80)
    print("OCR RAW OUTPUT:")
    print("="*80)
    print(ocr_text[:500])  # First 500 chars
    print("="*80 + "\n")
    
    return ocr_text


def extract_json_from_response(response_text: str) -> dict:
    """
    Extracts JSON from a response that may contain extra text/reasoning.
    Handles <think> tags, markdown, and other text before/after JSON.
    """
    
    # Step 1: Remove thinking tags and markdown
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = re.sub(r'//.*?(?=\n|,|]|})', '', cleaned)
    
    # Step 2: Find the first { and match it with its closing }
    # This is more reliable than trying to find multiple JSONs
    first_brace = cleaned.find('{')
    
    if first_brace == -1:
        raise ValueError("No JSON object found in response")
    
    # Count braces from the first { to find the matching }
    brace_count = 0
    start_idx = first_brace
    end_idx = -1
    
    for i in range(start_idx, len(cleaned)):
        char = cleaned[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        raise ValueError("Could not find matching closing brace for JSON object")
    
    # Extract and parse the JSON
    json_str = cleaned[start_idx:end_idx]
    
    try:
        parsed = json.loads(json_str)
        # Clean up numeric strings
        cleaned_data = _clean_numeric_values(parsed)
        # Validate it has required fields
        if 'invoice_number' in cleaned_data and 'patient_name' in cleaned_data:
            return cleaned_data
        else:
            raise ValueError("JSON missing required fields: invoice_number or patient_name")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}\nJSON string: {json_str[:200]}")



def _clean_numeric_values(data):
    """
    Recursively clean numeric string values in amount-related fields.
    Remove all non-numeric characters except decimal point from amounts.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Clean numeric for amount-related keys
            if k in ('amount', 'total_claimed_amount') and isinstance(v, str):
                import re as regex_module
                cleaned = regex_module.sub(r'[^\d.]', '', v.strip())
                if cleaned:
                    try:
                        result[k] = float(cleaned)
                    except ValueError:
                        result[k] = v
                else:
                    result[k] = v
            else:
                result[k] = _clean_numeric_values(v)
        return result
    elif isinstance(data, list):
        return [_clean_numeric_values(item) for item in data]
    elif isinstance(data, (int, float)):
        return float(data)
    else:
        return data


def _validate_claim_amounts(data: dict) -> tuple[bool, str]:
    """
    Validate that extracted amounts make sense.
    Catches OCR errors like 350.00 being read as 35000.
    Returns (is_valid, error_message)
    """
    try:
        total = data.get('total_claimed_amount', 0)
        line_items = data.get('line_items', [])
        
        if not line_items:
            return True, ""  # No line items to validate
        
        # Calculate sum of line items
        line_sum = sum(float(item.get('amount', 0)) for item in line_items)
        
        # Check 1: Total should roughly match sum of line items
        # Allow up to 10% difference for tax, fees, or rounding
        if line_sum > 0:
            difference_percent = abs(total - line_sum) / line_sum * 100
            if difference_percent > 10:  # 10% threshold instead of 5%
                return False, f"Total ${total:.2f} doesn't match line items sum ${line_sum:.2f} (diff: {difference_percent:.1f}%)"
        
        # Check 2: No single line item should be larger than total
        max_item = max((float(item.get('amount', 0)) for item in line_items), default=0)
        if max_item > total and total > 0:
            return False, f"Line item ${max_item:.2f} exceeds total ${total:.2f}"
        
        # Check 3: Detect suspiciously high amounts for simple services
        if line_items and total > 5000:
            desc = line_items[0].get('description', '').lower()
            simple_keywords = ['consultation', 'visit', 'exam', 'checkup', 'general', 'office', 'check up']
            
            if any(kw in desc for kw in simple_keywords):
                # Simple services typically cost $50-$2000
                return False, f"Amount ${total:.2f} seems too high for '{desc}' (typical range: $50-$2000)"
        
        return True, ""
        
    except Exception as e:
        return True, ""  # If validation fails, let it through


def _suggest_corrected_amount(extracted_amount: float) -> float:
    """
    If amount seems wrong, try to fix common OCR errors.
    e.g., 35000 might be 350.00 (misread decimal)
    """
    # If amount is suspiciously round (no cents) and > 1000
    if extracted_amount > 1000 and extracted_amount % 1 == 0:
        # Try dividing by 100 (might be missing decimal)
        suggested = extracted_amount / 100
        if 50 <= suggested <= 5000:  # Reasonable medical service cost
            return suggested
    
    return extracted_amount


_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Extract medical claim data from this billing invoice text.

INSTRUCTIONS:
1. Extract invoice_number: Look for "INVOICE", "Invoice Number", "Invoice #" etc.
2. Extract patient_name: Look for "PATIENT", "Name:", "Patient:"
3. Extract total_claimed_amount: Look for "TOTAL", "Total Due", "Amount Due", "Total Charges"
   - IMPORTANT: If you cannot find an explicit total, calculate it by SUMMING all line item amounts
   - Do NOT make up or guess a total
4. Extract line_items: List each service/item with description and amount
   - Look for rows with service descriptions and corresponding costs
   - Include amount for each line item

CRITICAL: 
- Only extract information that is explicitly shown in the text
- If a total is not clearly stated, use the sum of line items
- Do NOT invent numbers
- All amounts must be actual numbers from the text

Here is the text to extract from:
{ocr_text}

Return as JSON with this structure:
{{
    "invoice_number": "...",
    "patient_name": "...",
    "total_claimed_amount": (number - MUST match sum of line items),
    "line_items": [
        {{"description": "...", "amount": (number)}}
    ]
}}"""
)


def get_extraction_chain():
    chat_model = ChatOllama(model=LLM_MODEL, temperature=0.1)
    parser = JsonOutputParser(pydantic_object=StructuredClaimData)
    return _EXTRACTION_PROMPT | chat_model | parser


@st.cache_resource
def get_raw_extraction_chain():
    """Get chain without parser for fallback extraction."""
    chat_model = ChatOllama(model=LLM_MODEL, temperature=0.1)
    return _EXTRACTION_PROMPT | chat_model

# --- MAIN APP ---
def main():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    
    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.title("🛡️ 2026 Enterprise Claims Adjuster")
    
    # --- (NEW) SIDEBAR: POLICY MANAGEMENT ---
    with st.sidebar:
        st.header("📝 Policy Management")
        uploaded_policy = st.file_uploader("Upload New Policy (PDF or TXT)", type=["pdf", "txt"])
        
        if uploaded_policy:
            if st.button("Index Policy into AI Brain"):
                # Save uploaded file to local disk temporarily
                save_path = f"uploads/{uploaded_policy.name}"
                os.makedirs("uploads", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_policy.getbuffer())
                
                # Use our new function from rag_engine
                with st.spinner("Indexing policy..."):
                    success = add_policy_to_system(save_path)
                    if success:
                        st.success(f"Successfully indexed {uploaded_policy.name}!")
                    else:
                        st.error("Failed to index policy.")
    
    st.sidebar.info(f"Session Thread ID: {st.session_state.thread_id}")

    # --- 1. THE ADJUSTER INTERFACE (Human-in-the-loop) ---
    state = app.get_state(thread_config)

    # Check if the graph is currently interrupted (at the auditor node)
    if state.next:
        # Access the interrupt information sent from the auditor_node
        # Note: In LangGraph 1.x, the interrupt data is usually in the first value of state.next
        st.warning("⚠️ CLINICAL INTERRUPT: AI flagged this claim for Manual Review.")
        
        # Display the Critical Alerts from the state so the human knows what to look for
        if "critical_alerts" in state.values:
            with st.expander("🚨 AI Validation Alerts", expanded=True):
                for alert in state.values["critical_alerts"]:
                    st.error(alert)

        # Dashboard for the Human Adjuster
        with st.container(border=True):
            st.subheader("Adjuster Decision Desk")
            adj_notes = st.text_area("Adjustment Notes", placeholder="e.g., Verified medical necessity manually.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve Claim", type="primary", use_container_width=True):
                    # We send a dictionary back because auditor_node expects decision.get("action")
                    app.invoke(Command(resume={"action": "APPROVED", "notes": adj_notes}), thread_config)
                    st.rerun()
            with col2:
                if st.button("❌ Deny Claim", use_container_width=True):
                    app.invoke(Command(resume={"action": "DENIED", "notes": adj_notes}), thread_config)
                    st.rerun()
        st.stop() # Stop execution here so we don't show the "Upload" UI while a review is pending

    # --- 2. UPLOAD & INGESTION UI ---
    uploaded_file = st.file_uploader("Upload Medical Bill", type=["jpg", "png", "jpeg"])

    if uploaded_file and st.button("Start Autonomous Processing"):
        with st.spinner("Executing Agent Workflow..."):
            # Step 1: OCR + Extraction (Phase 1)
            with open(TEMP_FILE_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            raw_text = extract_text_from_bill(TEMP_FILE_PATH)
            extraction_chain = get_extraction_chain()

            try:
                # Try the chain first (with parser)
                structured_data = extraction_chain.invoke({"ocr_text": raw_text})
                st.success("Phase 1: Structured Data Extracted Successfully.")
            except Exception as parse_error:
                # If parsing fails, try to extract JSON from raw LLM response
                st.warning("Parser failed, attempting to extract JSON from raw response...")
                try:
                    raw_chain = get_raw_extraction_chain()
                    raw_response = raw_chain.invoke({"ocr_text": raw_text})
                    response_text = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)

                    structured_data = extract_json_from_response(response_text)
                    structured_data = StructuredClaimData.model_validate(structured_data).model_dump()
                    st.success("Phase 1: Structured Data Extracted Successfully (with fallback extraction).")
                except Exception as fallback_error:
                    st.error(f"Extraction Failed in Phase 1: {parse_error}\n\nFallback also failed: {fallback_error}")
                    st.stop()
            
            # DEBUG: Show what was extracted
            print("\n" + "="*80)
            print("LLM EXTRACTED DATA:")
            print("="*80)
            print(f"Invoice: {structured_data.get('invoice_number')}")
            print(f"Patient: {structured_data.get('patient_name')}")
            print(f"Total Amount: ${structured_data.get('total_claimed_amount')}")
            print(f"Line Items: {len(structured_data.get('line_items', []))} items")
            for i, item in enumerate(structured_data.get('line_items', []), 1):
                print(f"  Item {i}: {item.get('description')} = ${item.get('amount')}")
            print("="*80 + "\n")
            
            # Now let's verify by searching OCR text for the amount
            print("\n" + "="*80)
            print("AMOUNT VERIFICATION:")
            print("="*80)
            extracted_total = structured_data.get('total_claimed_amount', 0)
            print(f"Extracted total: ${extracted_total}")
            
            # Search OCR for similar numbers
            import re as search_re
            # Look for currency amounts in OCR
            amounts = search_re.findall(r'\$?\s*\d+[\.,]\d{2}|\$?\s*\d+(?:\,\d{3})*', raw_text)
            print(f"All amounts found in OCR text: {amounts}")
            
            # Look for the specific digits
            digits_extracted = search_re.sub(r'[^\d]', '', str(extracted_total))
            if digits_extracted in raw_text:
                context = raw_text.find(digits_extracted)
                print(f"✅ Found '{digits_extracted}' in OCR at position {context}")
                print(f"   Context: ...{raw_text[max(0, context-50):context+50]}...")
            else:
                print(f"❌ '{digits_extracted}' NOT found in OCR text")
                print(f"   This suggests OCR might have read something different")
            print("="*80 + "\n")

            # Step 2: Validate extracted amounts (catch OCR errors like 350->35000)
            is_valid, error_msg = _validate_claim_amounts(structured_data)
            
            if not is_valid:
                st.warning(f"⚠️ Amount Validation Issue: {error_msg}")
                st.info("The system detected a potential data quality issue.")
                
                # Try to auto-correct the amount
                corrected = _suggest_corrected_amount(structured_data.get('total_claimed_amount', 0))
                if corrected != structured_data.get('total_claimed_amount', 0):
                    st.success(f"✅ Auto-corrected: ${structured_data['total_claimed_amount']} → ${corrected:.2f}")
                    structured_data['total_claimed_amount'] = corrected
                else:
                    # Cannot auto-correct - offer options instead of blocking
                    st.warning("⚠️ Could not auto-correct. Showing options:")
                    
                    # Calculate sum of line items as suggested total
                    line_sum = sum(item.get('amount', 0) for item in structured_data.get('line_items', []))
                    
                    # Use session state to track user choice
                    if 'amount_resolution' not in st.session_state:
                        st.session_state.amount_resolution = None
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Use Extracted Amount", use_container_width=True, key="btn_extracted"):
                            st.session_state.amount_resolution = "extracted"
                            st.rerun()
                    
                    with col2:
                        if st.button("📊 Use Line Items Sum", use_container_width=True, key="btn_line_sum"):
                            st.session_state.amount_resolution = "line_sum"
                            st.rerun()
                    
                    with col3:
                        if st.button("✏️ Manual Entry", use_container_width=True, key="btn_manual"):
                            st.session_state.amount_resolution = "manual"
                            st.rerun()
                    
                    # Handle the user's choice
                    if st.session_state.amount_resolution == "extracted":
                        st.success(f"✅ Using extracted amount: ${structured_data['total_claimed_amount']:.2f}")
                        # Continue processing (don't stop)
                    
                    elif st.session_state.amount_resolution == "line_sum":
                        st.success(f"✅ Using line items sum: ${line_sum:.2f}")
                        structured_data['total_claimed_amount'] = line_sum
                        # Continue processing (don't stop)
                    
                    elif st.session_state.amount_resolution == "manual":
                        st.info("📝 Enter the correct total amount:")
                        manual_amount = st.number_input(
                            "Total amount:",
                            value=float(line_sum),
                            min_value=0.0,
                            step=0.01,
                            key="manual_amount_input"
                        )
                        
                        if st.button("✅ Confirm Amount", use_container_width=True, key="btn_confirm"):
                            st.session_state.amount_resolution = "manual_confirmed"
                            structured_data['total_claimed_amount'] = manual_amount
                            st.success(f"✅ Amount set to: ${manual_amount:.2f}")
                            st.rerun()
                        else:
                            st.stop()  # Wait for user to confirm
                    
                    # If no choice made yet, stop and wait
                    if st.session_state.amount_resolution not in ["extracted", "line_sum", "manual_confirmed"]:
                        st.stop()  # Wait for user to click a button
            
            # Step 3: Initialize Agentic Workflow (Phase 2)
            # We must provide all keys required by EnterpriseState
            initial_state = {
                "claim_id": str(uuid.uuid4()),
                "structured_data": structured_data,
                "validation_reports": [],
                "is_data_valid": True,
                "critical_alerts": [],
                "clinical_report": "",
                "policy_check_result": "Pending...",
                "audit_findings": "",
                "correction_needed": False,
                "final_verdict": "PROCESSING"
            }

            # Run the workflow
            # If it hits an 'interrupt', it will save state and return here.
            app.invoke(initial_state, thread_config)
            st.rerun() # Refresh to trigger the 'Adjuster Interface' UI above

    # --- 3. FINAL RESULTS VIEW ---
    # This only shows if the graph reaches 'END'
    if state.values and not state.next:
        st.header("Final Processing Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            verdict = state.values.get('final_verdict', 'UNKNOWN')
            if verdict == "APPROVED": st.success(f"### {verdict}")
            else: st.error(f"### {verdict}")
        
        with c2:
            st.metric("Claim Total", f"${state.values['structured_data'].get('total_claimed_amount', 0)}")
        
        with c3:
            # Only show "Valid" if validation explicitly passed
            # If workflow didn't update is_data_valid, show as "Unknown"
            status = state.values.get('is_data_valid')
            if status is True:
                st.metric("Clinical Status", "✅ Valid")
            elif status is False:
                st.metric("Clinical Status", "❌ Mismatched")
            else:
                st.metric("Clinical Status", "⚠️ Unknown")

        with st.expander("Full Audit Trace", expanded=True):
            st.write("**Policy Logic:**", state.values.get('policy_check_result'))
            st.write("**Human Notes:**", state.values.get('human_notes', 'None'))
            
            # Only show validation_reports if they're not empty
            validation_reports = state.values.get('validation_reports', [])
            if validation_reports:
                st.write("**Validation Reports:**")
                st.json(validation_reports)
            else:
                st.info("No validation reports available")

if __name__ == "__main__":
    main()