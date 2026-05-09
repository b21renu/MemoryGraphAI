"""
evaluation_metrics.py
---------------------
Complete evaluation suite for Graph RAG vs Vector RAG.

Two categories of metrics:

1. RANKING METRICS (operate on retrieved entity lists)
   These measure the quality of the retrieval step independent of the LLM.
   They prove that graph traversal surfaces more relevant entities.

2. LLM-AS-JUDGE METRICS (the gold standard for RAG evaluation)
   An LLM evaluator (Groq/Anthropic) rates each answer on three dimensions:
     - Relevance    [1-5]: Does the answer address what was asked?
     - Faithfulness [1-5]: Is the answer grounded in the provided context?
     - Completeness [1-5]: Does the answer cover all key aspects?
   
   Why LLM-as-judge is the RIGHT metric here:
   - It directly measures answer QUALITY, not just entity retrieval
   - It captures the benefit of relational context (Graph RAG's LLM gets richer
     structural evidence → produces more faithful, complete answers)
   - It is the standard used in production RAG evaluation (RAGAS, TruLens, etc.)
   - It is robust to the small graph size — unlike Recall which is bounded by K

3. CONTEXT QUALITY METRIC
   Context Informativeness Score: For each system, we ask the judge LLM
   whether the provided context ALONE is sufficient to answer the question.
   This directly quantifies the "is graph context better than flat entity names?" claim.
"""

import math
import json
import re
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ──────────────────────────────────────────────────────────────────────────────
 
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for x in top_k if x in relevant) / k
 
 
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for x in top_k if x in relevant) / len(relevant)
 
 
def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0
 
 
def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(retrieved, 1):
        if x in relevant:
            return 1.0 / i
    return 0.0
 
 
def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, x in enumerate(retrieved, 1):
        if x in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)
 
 
def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(items):
        return sum(1.0 / math.log2(i + 2) for i, x in enumerate(items) if x in relevant)
    actual = dcg(retrieved[:k])
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return actual / ideal if ideal else 0.0
 
 
def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0
 
 
def entity_coverage(retrieved: list[str], relevant: set[str]) -> float:
    """
    Fraction of relevant entities found ANYWHERE in the full retrieved list
    (not just top-k). Graph RAG returns more entities via traversal, so this
    metric shows whether traversal expands coverage beyond what vector search alone achieves.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge scoring
# ──────────────────────────────────────────────────────────────────────────────
 
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
 
You will evaluate an AI-generated answer on three criteria. Score each from 1-5.
 
QUESTION: {question}
 
GROUND TRUTH ANSWER: {ground_truth}
 
GENERATED ANSWER: {generated_answer}
 
Scoring rules:
- PENALISE answers that introduce facts NOT in the ground truth (hallucination). An answer that invents plausible-sounding but wrong details should score 1-2 on Faithfulness.
- REWARD answers that name specific entities or relationships that match the ground truth, even if phrasing differs.
- A SHORT, ACCURATE answer scores higher than a LONG, HALLUCINATED answer.
 
Evaluate on:
1. RELEVANCE [1-5]: Does the answer directly address the question asked?
   1=Completely off-topic, 3=Partially relevant, 5=Fully addresses the question
 
2. FAITHFULNESS [1-5]: Is the answer factually accurate with no invented details?
   1=Contains hallucinations or major errors, 3=Mostly correct with minor issues, 5=Completely accurate and grounded
 
3. COMPLETENESS [1-5]: Does the answer cover the key aspects from the ground truth?
   1=Misses most key points, 3=Covers main point but misses details, 5=Comprehensive coverage
 
Respond with ONLY a JSON object in this exact format (no other text):
{{"relevance": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>, "reasoning": "<one sentence>"}}"""
 
 
CONTEXT_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for RAG systems.
 
Given the QUESTION and the RETRIEVED CONTEXT provided to an AI assistant, evaluate whether the context is sufficient and relevant.
 
QUESTION: {question}
 
RETRIEVED CONTEXT:
{context}
 
Score the context on:
1. SUFFICIENCY [1-5]: Does the context contain enough information to answer the question?
   1=Context is useless, 3=Context partially helps, 5=Context fully enables a correct answer
 
2. PRECISION [1-5]: Is the context focused and relevant (low noise)?
   1=Mostly irrelevant information, 3=Mix of relevant and irrelevant, 5=All information is highly relevant
 
3. STRUCTURAL_RICHNESS [1-5]: Does the context reveal HOW concepts are connected, not just WHAT they are?
   1=Just entity names with no relationships, 3=Some relationship hints, 5=Clear relationship chains that enable reasoning
 
Respond with ONLY a JSON object (no other text):
{{"sufficiency": <1-5>, "precision": <1-5>, "structural_richness": <1-5>, "reasoning": "<one sentence>"}}"""
 
 
def parse_judge_response(response_text: str, expected_keys: list[str]) -> dict:
    """Extract JSON scores from LLM judge response, with fallback parsing."""
    # Try direct JSON parse
    try:
        # Find JSON object in response
        match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if all(k in data for k in expected_keys):
                return data
    except (json.JSONDecodeError, AttributeError):
        pass
 
    # Fallback: look for numeric scores in text
    scores = {}
    for key in expected_keys:
        if key == "reasoning":
            scores[key] = "Could not parse reasoning"
            continue
        pattern = rf'["\']?{key}["\']?\s*[:\s]+(\d)'
        match = re.search(pattern, response_text, re.IGNORECASE)
        scores[key] = int(match.group(1)) if match else 3  # default to middle score
 
    return scores
 
 
def llm_judge_answer(
    question: str,
    ground_truth: str,
    generated_answer: str,
    llm_fn,
) -> dict:
    """
    Uses the LLM as a judge to score a generated answer.
    Returns dict with relevance, faithfulness, completeness scores (1-5).
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )
    response = llm_fn(prompt)
    scores = parse_judge_response(response, ["relevance", "faithfulness", "completeness", "reasoning"])
    # Normalise to [0,1] range for consistency with other metrics
    for key in ["relevance", "faithfulness", "completeness"]:
        if key in scores and isinstance(scores[key], (int, float)):
            scores[f"{key}_norm"] = (scores[key] - 1) / 4.0  # 1→0.0, 5→1.0
    return scores
 
 
def llm_judge_context(
    question: str,
    context: str,
    llm_fn,
) -> dict:
    """
    Uses the LLM to evaluate the quality of the retrieved context itself.
    This is the key metric: does graph context provide better structural evidence?
    Returns dict with sufficiency, precision, structural_richness scores (1-5).
    """
    prompt = CONTEXT_JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context[:1500],  # truncate to avoid token limit
    )
    response = llm_fn(prompt)
    scores = parse_judge_response(
        response, ["sufficiency", "precision", "structural_richness", "reasoning"]
    )
    for key in ["sufficiency", "precision", "structural_richness"]:
        if key in scores and isinstance(scores[key], (int, float)):
            scores[f"{key}_norm"] = (scores[key] - 1) / 4.0
    return scores
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────
 
def compute_ranking_metrics(
    retrieved: list[str],
    relevant: set[str],
    k: int = 5,
    latency: float = 0.0,
) -> dict:
    return {
        "Precision@K":      round(precision_at_k(retrieved, relevant, k), 4),
        "Recall@K":         round(recall_at_k(retrieved, relevant, k), 4),
        "F1@K":             round(f1_at_k(retrieved, relevant, k), 4),
        "MRR":              round(mrr(retrieved, relevant), 4),
        "MAP":              round(average_precision(retrieved, relevant), 4),
        "nDCG@K":           round(ndcg_at_k(retrieved, relevant, k), 4),
        "HitRate@K":        round(hit_rate_at_k(retrieved, relevant, k), 4),
        "EntityCoverage":   round(entity_coverage(retrieved, relevant), 4),
        "Latency(s)":       round(latency, 4),
    }