import os
from dotenv import load_dotenv
from typing import Dict
from langgraph.graph import StateGraph, END
from app.models.models import DocRecord, Profile, MatchScore
from app.helpers.parsing import load_folder
from app.utils.utils import ollama_generate, safe_json
from app.helpers.prompts import EXTRACT_PROMPT, CONSISTENCY_PROMPT
from app.services.matching import rank_and_score, categorize
import pandas as pd
from pathlib import Path
from typing import Any, List

load_dotenv()
CV_DIR = os.getenv("CV_DIR", "./data/cvs")
JD_DIR = os.getenv("JD_DIR", "./data/jds")
REPORT_DIR = os.getenv("REPORT_DIR", "./reports")
SELECT_MIN = float(os.getenv("SELECT_MIN", "0.72"))
REJECT_MAX = float(os.getenv("REJECT_MAX", "0.48"))

def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        # join list of sentences or tokens into one paragraph
        return " ".join([str(t).strip() for t in x if str(t).strip()])
    return str(x).strip()

def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        # split on commas/semicolons; normalize tokens
        parts = [p.strip() for p in x.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    return []


def extract_profile(doc: DocRecord) -> Profile:
    # Ask LLM
    resp = ollama_generate(EXTRACT_PROMPT.format(doc=doc.text))
    data = safe_json(resp, fallback={})

    # Coerce all fields to correct types
    name = _as_text(data.get("name"))
    email = _as_text(data.get("email"))
    phone = _as_text(data.get("phone"))

    years_experience = data.get("years_experience")
    try:
        if isinstance(years_experience, list):
            # sometimes model returns ["6"]; take first
            years_experience = years_experience[0] if years_experience else None
        years_experience = float(years_experience) if years_experience not in (None, "") else None
    except Exception:
        years_experience = None

    roles = _as_list(data.get("roles"))
    skills = [s.lower() for s in _as_list(data.get("skills"))]
    tools = [s.lower() for s in _as_list(data.get("tools"))]
    domains = [s.lower() for s in _as_list(data.get("domains"))]
    certs = _as_list(data.get("certifications"))

    summary = _as_text(data.get("summary"))
    if not summary:
        # fallback to first 1000 chars of raw text
        summary = doc.text[:1000]

    return Profile(
        name=name or None,
        email=email or None,
        phone=phone or None,
        years_experience=years_experience,
        roles=roles,
        skills=skills,
        tools=tools,
        domains=domains,
        certifications=certs,
        summary=summary
    )

def llm_consistency(jd: Profile, cv: Profile):
    resp = ollama_generate(CONSISTENCY_PROMPT.format(
        jd_summary=jd.summary,
        jd_skills=", ".join(jd.skills[:40]),
        cv_summary=cv.summary,
        cv_skills=", ".join(cv.skills[:40]),
    ))
    data = safe_json(resp, {"score": 0.5, "why": "Heuristic fallback"})
    s = float(data.get("score", 0.5))
    why = data.get("why", "")
    return max(0.0, min(1.0, s)), why

def write_reports(jdid: str, rows: Dict[str, MatchScore], jd_prof: Profile):
    from pathlib import Path
    import os

    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # Build DataFrame safely
    data = [{
        "jd_id": s.jd_id,
        "cv_id": s.cv_id,
        "category": s.category,
        "total_score": round(s.total_score, 4),
        "sim_embed": round(s.sim_embed, 4),
        "skill_coverage": round(s.skill_coverage, 4),
        "must_have_penalty": round(s.must_have_penalty, 4),
        "llm_consistency": round(s.llm_consistency, 4),
        "rationale": s.rationale
    } for s in rows.values()] if rows else []

    df = pd.DataFrame(data, columns=[
        "jd_id","cv_id","category","total_score","sim_embed",
        "skill_coverage","must_have_penalty","llm_consistency","rationale"
    ])

    csv_path = os.path.join(REPORT_DIR, f"{jdid}_report.csv")
    if len(df):
        df.sort_values("total_score", ascending=False).to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)  # empty file with headers

    # Markdown (top-10 or note no matches)
    md_lines = [f"# JD {jdid} — Top Matches"]
    jd_summary = (jd_prof.summary or "").strip() if jd_prof else ""
    if jd_summary:
        md_lines.append(f"**JD Summary**: {jd_summary}\n")
    else:
        md_lines.append("*(No JD summary available)*\n")

    if len(df):
        top_md = df.sort_values("total_score", ascending=False).head(10)
        md_lines += [
            "| Rank | CV ID | Category | Score | Coverage | EmbedSim | LLM | Penalty |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
        for i, r in enumerate(top_md.itertuples(), start=1):
            md_lines.append(
                f"| {i} | {r.cv_id} | {r.category} | {r.total_score:.3f} | "
                f"{r.skill_coverage:.3f} | {r.sim_embed:.3f} | {r.llm_consistency:.3f} | {r.must_have_penalty:.3f} |"
            )
        md_lines.append("\n---\nRationales (top-5):")
        for r in df.sort_values("total_score", ascending=False).head(5).itertuples():
            md_lines.append(f"- **{r.cv_id}**: {r.rationale}")
    else:
        md_lines.append("> No candidates matched this JD.\n")

    md_path = os.path.join(REPORT_DIR, f"{jdid}_top.md")
    Path(md_path).write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path

# LangGraph state and nodes
class S(dict):
    pass

def node_parse(state: S):
    # ensure folders
    for p in [CV_DIR, JD_DIR]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Folder not found: {p}. Update CV_DIR/JD_DIR in .env")
    cvs = load_folder(CV_DIR)
    jds = load_folder(JD_DIR)
    return {"cv_docs": cvs, "jd_docs": jds}  # DELTA

def node_extract(state: S):
    cv_docs = state.get("cv_docs", [])
    jd_docs = state.get("jd_docs", [])

    cv_map = {}
    for d in cv_docs:
        try:
            cv_map[d.id] = extract_profile(d)
        except Exception as e:
            # minimal fallback so matching can proceed
            cv_map[d.id] = Profile(summary=d.text[:1000])
            print(f"[WARN] CV extract failed for {d.path}: {e}")

    jd_map = {}
    for d in jd_docs:
        try:
            jd_map[d.id] = extract_profile(d)
        except Exception as e:
            jd_map[d.id] = Profile(summary=d.text[:1000])
            print(f"[WARN] JD extract failed for {d.path}: {e}")

    return {"cv_profiles": cv_map, "jd_profiles": jd_map}

def node_match(state: S):
    cv_profiles = state.get("cv_profiles", {})
    jd_profiles = state.get("jd_profiles", {})
    results = {}
    for jdid, jd in jd_profiles.items():
        musts = [s for s in jd.skills if "must" in s.lower()]
        weights = {s: 0.5 for s in jd.skills[:10]}
        scored = rank_and_score(jdid, jd, cv_profiles, musts, weights, llm_consistency)
        for sc in scored:
            sc.category = categorize(sc.total_score, SELECT_MIN, REJECT_MAX)
        results[jdid] = scored
    return {"matches": results}  # DELTA

def node_report(state: S):
    jd_profiles = state.get("jd_profiles", {})
    report_paths = []
    for jdid, scores in state.get("matches", {}).items():
        jd_prof = jd_profiles.get(jdid) or Profile(summary="")
        csv, md = write_reports(jdid, {f"{s.cv_id}": s for s in scores}, jd_prof)
        report_paths.append((jdid, csv, md))
    return {"report_paths": report_paths}  # return delta

def build_graph():
    g = StateGraph(S)
    g.add_node("parse", node_parse)
    g.add_node("extract", node_extract)
    g.add_node("match", node_match)
    g.add_node("report", node_report)
    g.set_entry_point("parse")
    g.add_edge("parse", "extract")
    g.add_edge("extract", "match")
    g.add_edge("match", "report")
    g.add_edge("report", END)
    return g.compile()

def run_sequential():
    # parse
    parse_out = node_parse({})
    # extract
    extract_out = node_extract({**parse_out})
    # match
    match_out = node_match({**parse_out, **extract_out})
    # report
    report_out = node_report({**parse_out, **extract_out, **match_out})
    # final merged dict
    return {**parse_out, **extract_out, **match_out, **report_out}
