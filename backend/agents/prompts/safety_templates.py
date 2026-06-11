"""
Hardcoded safety templates injected by guardrails — never LLM-generated.
These are legal-grade strings; treat them as constants, not suggestions.
"""

# Appended after Tier 2 answers
REGULATORY_DISCLAIMER = (
    "Please keep in mind that federal financial aid rules can change, and your situation "
    "may have details that affect the outcome. I strongly recommend confirming this with "
    "your school's financial aid office or visiting StudentAid.gov for the most current guidance."
)

# Appended when retrieval confidence is below threshold
LOW_CONFIDENCE_DISCLAIMER = (
    "I want to make sure you have accurate information. My answer is based on general "
    "federal guidelines, but I recommend verifying this directly at StudentAid.gov or "
    "speaking with your financial aid office."
)

# Full response for Tier 3 escalation — replaces any LLM answer
ESCALATION_RESPONSES = {
    "default": (
        "That's a great question, but it really depends on your specific situation — things like "
        "your income, family size, enrollment status, and school policies all play a role. "
        "To get an accurate answer, I'd recommend speaking directly with your school's financial "
        "aid office. They can review your file and give you a personalized answer. You can also "
        "visit StudentAid.gov or call the Federal Student Aid Information Center at "
        "1-800-433-3243 — they're available Monday through Friday."
    ),
    "eligibility": (
        "Determining whether you personally qualify for a specific aid program requires reviewing "
        "your complete financial and enrollment information — something I'm not able to do on "
        "this call. Your financial aid office can run that analysis for you. I'd suggest "
        "contacting them directly or logging into StudentAid.gov to review your FAFSA summary."
    ),
    "appeal": (
        "Deciding whether to appeal a financial aid decision is something your financial aid "
        "office is best positioned to advise on — they know your full file and what documentation "
        "would support a successful appeal. I'd reach out to them directly. You can also find "
        "general information about the appeal process on StudentAid.gov."
    ),
    "loan_forgiveness": (
        "Loan forgiveness eligibility is highly specific to your loan type, repayment plan, "
        "employer, and payment history. I can share general program information, but to know "
        "if you personally qualify, you'll need to contact your loan servicer or visit "
        "StudentAid.gov's loan forgiveness section."
    ),
}


def get_escalation_response(reason: str | None = None) -> str:
    """Return the most appropriate escalation response based on context."""
    if not reason:
        return ESCALATION_RESPONSES["default"]
    reason_lower = reason.lower()
    if "eligib" in reason_lower or "qualify" in reason_lower:
        return ESCALATION_RESPONSES["eligibility"]
    if "appeal" in reason_lower:
        return ESCALATION_RESPONSES["appeal"]
    if "forgiveness" in reason_lower or "forgive" in reason_lower:
        return ESCALATION_RESPONSES["loan_forgiveness"]
    return ESCALATION_RESPONSES["default"]
