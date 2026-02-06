from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from langchain_groq import ChatGroq

from medical_tools import lookup_clinical_codes
from rag_engine import get_policy_context
from dotenv import load_dotenv

load_dotenv()


class EnterpriseState(TypedDict):
    structured_data: dict
    clinical_report: str
    policy_check_result: str
    audit_findings: str
    correction_needed: bool
    final_verdict: str
    is_data_valid: bool  # Added - for clinical status display
    validation_reports: list  # Added - for validation details
    critical_alerts: list  # Added - for human adjuster alerts
    claim_id: str  # Added - for tracking


# Single place to configure the LLM used for policy checking
power_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# --- NODES ---
def clinical_validator(state: EnterpriseState):
    """Agent: Maps descriptions to real medical codes and validates claim data."""
    try:
        items = state["structured_data"]["line_items"]
        total = state["structured_data"].get("total_claimed_amount", 0)
        
        # Validate amount makes sense
        line_sum = sum(item.get('amount', 0) for item in items)
        difference_percent = abs(total - line_sum) / line_sum * 100 if line_sum > 0 else 0
        
        validation_issues = []
        
        # Check if total matches line items
        if difference_percent > 10:
            validation_issues.append(f"Total ${total} doesn't match line items ${line_sum:.2f}")
            is_valid = False
        else:
            is_valid = True
        
        # Generate clinical codes report
        report = [f"Found: {lookup_clinical_codes(i['description'])}" for i in items]
        
        return {
            "clinical_report": "\n".join(report),
            "is_data_valid": is_valid,
            "validation_reports": validation_issues,
            "critical_alerts": validation_issues
        }
    except Exception as e:
        return {
            "clinical_report": f"Clinical validation error: {str(e)}",
            "is_data_valid": False,
            "validation_reports": [f"Error: {str(e)}"],
            "critical_alerts": [f"Error during clinical validation: {str(e)}"]
        }


def policy_checker_node(state: EnterpriseState):
    """Agent: Advanced Policy Checker using RAG from persistent vector store."""
    print("--- NODE: Advanced Policy Checker ---")

    # Extract the main service being claimed
    line_items = state["structured_data"].get("line_items", [])
    claimed_service = line_items[0]["description"] if line_items else "General Medical Visit"

    # Retrieve policy context from persistent ChromaDB vector store
    context = get_policy_context(claimed_service)

    # Logic Check with the LLM
    prompt = f"""
    POLICY CONTEXT:
    {context}

    CLAIM DATA:
    Service: {claimed_service}
    Amount: ${state["structured_data"].get("total_claimed_amount", 0)}

    TASK:
    Identify if this service is 'Covered', 'Excluded', or 'Requires Authorization' 
    based ONLY on the Policy Context above. Cite the specific Section ID or Page Number.
    """

    # Call the LLM
    response = power_llm.invoke(prompt)
    content = getattr(response, "content", str(response))

    return {"policy_check_result": content}


def auditor_node(state: EnterpriseState):
    """Agent: Final auditor that makes approve/deny decisions."""
    amt = state["structured_data"].get("total_claimed_amount", 0)
    policy_result = state.get("policy_check_result", "").lower()
    
    # DEBUG: Print what we're working with
    print(f"\n--- AUDITOR NODE DEBUG ---")
    print(f"Amount: ${amt}")
    print(f"Policy result preview: {policy_result[:100]}")
    
    # Check if policy clearly says "COVERED" or "APPROVED"
    has_covered = "covered" in policy_result
    has_excluded = "excluded" in policy_result
    has_approved = "approved" in policy_result
    has_automatic = "automatic" in policy_result
    
    print(f"Has 'covered': {has_covered}")
    print(f"Has 'excluded': {has_excluded}")
    print(f"Has 'approved': {has_approved}")
    print(f"Has 'automatic': {has_automatic}")
    
    # Service is approved if policy says so AND not excluded
    policy_approved = (has_covered and not has_excluded) or has_approved or has_automatic
    
    print(f"Policy approved: {policy_approved}")
    print(f"Amount check: {amt} < 25000 = {amt < 25000}")
    print(f"--- END DEBUG ---\n")
    
    # 2026 Business Rule: 
    # - If policy says EXCLUDED: Auto-deny
    # - If policy says COVERED/APPROVED and amount < $25,000: Auto-approve
    # - If amount >= $25,000: Require human review
    
    if has_excluded:
        # Excluded services: auto-deny
        return {
            "audit_findings": "Service is EXCLUDED from coverage per policy",
            "correction_needed": False,
            "final_verdict": "DENIED",
        }
    
    if policy_approved:
        if amt < 25000:
            # Covered service under $25k: auto-approve
            return {
                "audit_findings": f"Auto-Approved: Service covered per policy, amount ${amt:.2f} under $25,000 threshold",
                "correction_needed": False,
                "final_verdict": "APPROVED",
            }
        else:
            # Covered but high-value: require human review
            human_input = interrupt({"msg": f"High value claim (${amt:,.2f}). Approve?", "data": state})
            verdict = str(human_input).strip().upper() if human_input else "MANUAL_REVIEW"
            if verdict not in {"APPROVED", "DENIED"}:
                verdict = "MANUAL_REVIEW"
            return {
                "audit_findings": f"High-value covered service - Human Decision: {human_input}",
                "correction_needed": False,
                "final_verdict": verdict,
            }
    
    # Policy status unclear, require manual review
    return {
        "audit_findings": "Policy status unclear - requires manual review",
        "correction_needed": False,
        "final_verdict": "MANUAL_REVIEW",
    }


# --- THE ORCHESTRATOR ---
def _build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(EnterpriseState)
    workflow.add_node("validator", clinical_validator)
    workflow.add_node("policy_checker", policy_checker_node)
    workflow.add_node("auditor", auditor_node)

    workflow.set_entry_point("validator")
    workflow.add_edge("validator", "policy_checker")
    workflow.add_edge("policy_checker", "auditor")
    workflow.add_edge("auditor", END)

    # Persistent Memory is required for HITL to save the state while waiting
    memory = MemorySaver()
    compiled_app = workflow.compile(checkpointer=memory)
    
    return compiled_app


# Build the app at module initialization
app = _build_graph()

# Explicit exports for clarity
__all__ = ["app", "EnterpriseState"]