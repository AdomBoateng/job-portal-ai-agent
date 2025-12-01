from typing import Dict, List
import numpy as np
from app.models.models import Profile, MatchScore
from app.utils.utils import ollama_embed

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set([x.lower() for x in a]), set([x.lower() for x in b])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def weighted_skill_coverage(jd_skills: List[str], cv_skills: List[str], weights: Dict[str,float]=None) -> float:
    weights = weights or {}
    jd_set = set([s.lower() for s in jd_skills])
    cv_set = set([s.lower() for s in cv_skills])
    if not jd_set:
        return 0.0
    score = 0.0
    total = 0.0
    for s in jd_set:
        w = 1.0 + weights.get(s, 0.0)
        total += w
        if s in cv_set:
            score += w
    return score/total if total>0 else 0.0

def must_have_penalty(jd_skills: List[str], cv_skills: List[str], must_set: List[str]) -> float:
    # returns penalty 0..1 (higher penalty for missing must-haves)
    miss = 0
    ms = set([m.lower() for m in must_set])
    cvs = set([c.lower() for c in cv_skills])
    for m in ms:
        if m not in cvs:
            miss += 1
    return min(1.0, miss / max(1, len(ms))) if ms else 0.0

def embed_similarity(a: str, b: str) -> float:
    va = ollama_embed(a)
    vb = ollama_embed(b)
    num = float(np.dot(va, vb))
    den = float(np.linalg.norm(va)*np.linalg.norm(vb)) or 1e-8
    return max(0.0, min(1.0, num/den))

def categorize(total, smin=0.72, rmax=0.48):
    if total >= smin:
        return "SELECT"
    if total <= rmax:
        return "REJECT"
    return "NEED_FURTHER_EVAL"

def rank_and_score(
    jdid: str, jd: Profile, cv_map: Dict[str, Profile],
    must_skills: List[str], weights: Dict[str,float],
    llm_consistency_cb
) -> List[MatchScore]:

    out = []
    for cvid, cv in cv_map.items():
        sim = embed_similarity(jd.summary, cv.summary)
        cover = weighted_skill_coverage(jd.skills, cv.skills, weights)
        penalty = must_have_penalty(jd.skills, cv.skills, must_skills)
        llm_s, llm_why = llm_consistency_cb(jd, cv)

        total = (0.5*sim) + (0.35*cover) + (0.15*llm_s) - (0.2*penalty)
        out.append(MatchScore(
            jd_id=jdid, cv_id=cvid, sim_embed=sim, skill_coverage=cover,
            must_have_penalty=penalty, llm_consistency=llm_s,
            total_score=total, category="PENDING", rationale=llm_why
        ))
    return out
