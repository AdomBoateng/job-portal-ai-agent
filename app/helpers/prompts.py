EXTRACT_PROMPT = """You are an information extractor.
Given the document below (CV or JD), return strict JSON with keys:
name, email, phone, years_experience, roles, skills, tools, domains, certifications, summary.

- Normalize skills/tools/domains to short lowercase tokens.
- If unknown, use null or empty list.
- Keep summary to 3–5 concise sentences.

DOCUMENT:
{doc}
"""

CONSISTENCY_PROMPT = """You are a recruiter assistant. Score how well this CV matches the JD on a 0–1 scale.
Return JSON: {{"score": <0..1>, "why": "<1-2 sentences>"}}

JD SUMMARY:
{jd_summary}

JD KEY SKILLS:
{jd_skills}

CV SUMMARY:
{cv_summary}

CV KEY SKILLS:
{cv_skills}
"""
