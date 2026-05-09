"""
run_comparison.py
-----------------
Full head-to-head evaluation: Graph RAG vs Vector RAG.

Uses LLM-as-judge (Groq API) for gold-standard answer quality scoring,
plus traditional ranking metrics.

Run:
    python run_comparison.py
    
Outputs:
    comparison_results.json
    comparison_summary.csv
"""

import os
import sys
import json
import csv
import time
import math
 
# Make parent directory importable (for running from Comparison_Analysis_v3/)
sys.path.insert(0, os.path.dirname(__file__))
 
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
 
from knowledge_graph import get_graph
from embedder import build_embedder
from graph_rag_pipeline import GraphRAGPipeline
from vector_pipeline import VectorRAGPipeline
from evaluation_metrics import (
    compute_ranking_metrics,
    llm_judge_answer,
    llm_judge_context,
)
from eval_queries import EVAL_QUERIES
 
# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
 
K_GRAPH = 10   # Graph RAG retrieves seeds + neighbours — natural depth is larger
K_VECTOR = 5   # Vector RAG retrieves top-5 by cosine similarity
SEED_K = 5     # Number of ANN seed nodes before graph expansion
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LLM wrapper (Groq)
# ──────────────────────────────────────────────────────────────────────────────
 
def make_groq_llm(model_name: str = "llama-3.1-8b-instant"):
    """Returns a callable that takes a prompt string and returns a response string."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            temperature=0,
            model_name=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
        def call(prompt: str) -> str:
            return llm.invoke(prompt).content
        return call
    except Exception as e:
        print(f"  WARNING: Groq LLM unavailable ({e}). Using stub responses.")
        def stub(prompt: str) -> str:
            return "LLM unavailable — stub response."
        return stub
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main comparison loop
# ──────────────────────────────────────────────────────────────────────────────
 
def run_comparison():
    print("\n" + "=" * 76)
    print("  MemoryGraph RAG  vs  Vector RAG  — Full Evaluation  (v3)")
    print("  Self-contained graph · LLM-as-Judge · Hybrid Retrieval")
    print("=" * 76)
 
    # Build shared infrastructure
    graph = get_graph()
    print(f"\n  Knowledge graph: {len(graph['entities'])} entities, "
          f"{len(graph['relationships'])} relationships\n")
 
    embedder = build_embedder(graph)
 
    # TWO separate LLMs:
    # - answer_llm: stronger model for generating answers from graph/vector context
    #   llama-3.3-70b-versatile can reason over relational triples properly
    # - judge_llm:  lighter model sufficient for scoring structured JSON responses
    answer_llm = make_groq_llm("llama-3.3-70b-versatile")
    judge_llm  = make_groq_llm("llama-3.1-8b-instant")
    print("  LLMs: answer_generation=llama-3.3-70b-versatile  judge=llama-3.1-8b-instant\n")
 
    graph_pipeline = GraphRAGPipeline(embedder=embedder, seed_k=SEED_K)
    vector_pipeline = VectorRAGPipeline(embedder=embedder, top_k=K_VECTOR)
 
    print(f"  Evaluation design:")
    print(f"    Graph RAG  — seed_k={SEED_K} ANN seeds + graph traversal → evaluated at K={K_GRAPH}")
    print(f"    Vector RAG — top_k={K_VECTOR} cosine similarity           → evaluated at K={K_VECTOR}")
    print(f"  NOTE: Graph RAG is evaluated at K=10 because hybrid retrieval naturally returns")
    print(f"  more entities (seeds + traversal-found neighbours). Using K=5 for both would")
    print(f"  unfairly discard the graph-traversal benefit by truncating the ranked list.")
    print(f"  This is documented in the README and is by design, not a bug.\n")
 
    all_results = []
    judge_scores_graph = {"relevance": [], "faithfulness": [], "completeness": []}
    judge_scores_vector = {"relevance": [], "faithfulness": [], "completeness": []}
    ctx_scores_graph = {"sufficiency": [], "precision": [], "structural_richness": []}
    ctx_scores_vector = {"sufficiency": [], "precision": [], "structural_richness": []}
 
    for idx, item in enumerate(EVAL_QUERIES, start=1):
        query = item["query"]
        relevant = item["relevant_entities"]
        ground_truth = item["ground_truth"]
        q_type = item.get("query_type", "")
 
        print(f"\n[{idx}/{len(EVAL_QUERIES)}] {query}")
        print(f"  Type: {q_type}")
        print(f"  Relevant: {relevant}")
        print("-" * 64)
 
        # ── Graph RAG ──────────────────────────────────────────────────────
        t0 = time.time()
        graph_retrieved, graph_triples = graph_pipeline.retrieve(query)
        graph_context = graph_pipeline.get_context_string(query)
        graph_answer = graph_pipeline.answer_question(query, answer_llm)
        graph_latency = time.time() - t0
 
        graph_rank = compute_ranking_metrics(graph_retrieved, relevant, K_GRAPH, graph_latency)
        graph_judge = llm_judge_answer(query, ground_truth, graph_answer, judge_llm)
        graph_ctx_judge = llm_judge_context(query, graph_context, judge_llm)
 
        # ── Vector RAG ─────────────────────────────────────────────────────
        t1 = time.time()
        vector_retrieved, _ = vector_pipeline.search(query)
        vector_context = vector_pipeline.get_context_string(query)
        vector_answer = vector_pipeline.answer_question(query, answer_llm)
        vector_latency = time.time() - t1
 
        vector_rank = compute_ranking_metrics(vector_retrieved, relevant, K_VECTOR, vector_latency)
        vector_judge = llm_judge_answer(query, ground_truth, vector_answer, judge_llm)
        vector_ctx_judge = llm_judge_context(query, vector_context, judge_llm)
 
        # ── Accumulate judge scores ────────────────────────────────────────
        for key in ["relevance", "faithfulness", "completeness"]:
            judge_scores_graph[key].append(graph_judge.get(key, 3))
            judge_scores_vector[key].append(vector_judge.get(key, 3))
        for key in ["sufficiency", "precision", "structural_richness"]:
            ctx_scores_graph[key].append(graph_ctx_judge.get(key, 3))
            ctx_scores_vector[key].append(vector_ctx_judge.get(key, 3))
 
        # ── Print per-query table ──────────────────────────────────────────
        print(f"\n  Graph retrieved ({len(graph_retrieved)} entities, K={K_GRAPH}): {graph_retrieved[:10]}")
        print(f"  Vector retrieved ({len(vector_retrieved)} entities, K={K_VECTOR}): {vector_retrieved[:5]}")
        print(f"  GRAPH CONTEXT  ({len(graph_triples)} triples): "
              f"{graph_context[:120].replace(chr(10),' | ')}...")
        print()
        print(f"  {'Metric':<28} {'Graph RAG':>10} {'Vector RAG':>10}  {'Winner':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10}  {'-'*10}")
 
        # Ranking metrics
        for m in ["Precision@K", "Recall@K", "F1@K", "MRR", "MAP", "nDCG@K",
                  "HitRate@K", "EntityCoverage"]:
            gv, vv = graph_rank[m], vector_rank[m]
            w = "Graph ✓" if gv > vv else ("Vector ✓" if vv > gv else "Tie")
            print(f"  {m:<28} {gv:>10.4f} {vv:>10.4f}  {w:>10}")
 
        # LLM-as-judge answer scores
        for label, gv, vv in [
            ("Judge:Relevance[1-5]",    graph_judge.get("relevance",3),    vector_judge.get("relevance",3)),
            ("Judge:Faithfulness[1-5]", graph_judge.get("faithfulness",3), vector_judge.get("faithfulness",3)),
            ("Judge:Completeness[1-5]", graph_judge.get("completeness",3), vector_judge.get("completeness",3)),
        ]:
            w = "Graph ✓" if gv > vv else ("Vector ✓" if vv > gv else "Tie")
            print(f"  {label:<28} {gv:>10} {vv:>10}  {w:>10}")
 
        # Context judge scores
        for label, gv, vv in [
            ("Ctx:Sufficiency[1-5]",        graph_ctx_judge.get("sufficiency",3),        vector_ctx_judge.get("sufficiency",3)),
            ("Ctx:Precision[1-5]",          graph_ctx_judge.get("precision",3),          vector_ctx_judge.get("precision",3)),
            ("Ctx:StructuralRichness[1-5]", graph_ctx_judge.get("structural_richness",3),vector_ctx_judge.get("structural_richness",3)),
        ]:
            w = "Graph ✓" if gv > vv else ("Vector ✓" if vv > gv else "Tie")
            print(f"  {label:<28} {gv:>10} {vv:>10}  {w:>10}")
 
        print(f"  {'Latency(s)':<28} {graph_latency:>10.2f} {vector_latency:>10.2f}  "
              f"{'Graph ✓' if graph_latency <= vector_latency else 'Vector ✓':>10}")
 
        print(f"\n  Graph RAG reasoning: {graph_judge.get('reasoning','')}")
        print(f"  Vector RAG reasoning: {vector_judge.get('reasoning','')}")
        print(f"  Graph context quality: {graph_ctx_judge.get('reasoning','')}")
        print(f"  Vector context quality: {vector_ctx_judge.get('reasoning','')}")
 
        all_results.append({
            "query": query,
            "query_type": q_type,
            "relevant_entities": list(relevant),
            "graph_retrieved": graph_retrieved,
            "vector_retrieved": vector_retrieved,
            "graph_triples_count": len([l for l in graph_context.splitlines() if l.strip()]),
            "graph_context": graph_context,
            "vector_context": vector_context[:600],
            "graph_answer": graph_answer,
            "vector_answer": vector_answer,
            "graph_ranking_metrics": graph_rank,
            "vector_ranking_metrics": vector_rank,
            "graph_judge_scores": graph_judge,
            "vector_judge_scores": vector_judge,
            "graph_context_judge": graph_ctx_judge,
            "vector_context_judge": vector_ctx_judge,
        })

        time.sleep(0.5)
 
    # ── Macro averages ─────────────────────────────────────────────────────────
    n = len(EVAL_QUERIES)
 
    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
 
    ranking_keys = ["Precision@K","Recall@K","F1@K","MRR","MAP","nDCG@K","HitRate@K","EntityCoverage"]
    graph_rank_avg = {k: avg([r["graph_ranking_metrics"][k] for r in all_results]) for k in ranking_keys}
    vector_rank_avg = {k: avg([r["vector_ranking_metrics"][k] for r in all_results]) for k in ranking_keys}
 
    graph_judge_avg = {k: avg(judge_scores_graph[k]) for k in judge_scores_graph}
    vector_judge_avg = {k: avg(judge_scores_vector[k]) for k in judge_scores_vector}
    graph_ctx_avg = {k: avg(ctx_scores_graph[k]) for k in ctx_scores_graph}
    vector_ctx_avg = {k: avg(ctx_scores_vector[k]) for k in ctx_scores_vector}
 
    print("\n" + "=" * 76)
    print("  MACRO-AVERAGED RESULTS")
    print("=" * 76)
 
    graph_wins = 0
    all_metric_rows = []
 
    def print_row(label, gv, vv, higher_is_better=True):
        nonlocal graph_wins
        if higher_is_better:
            w = "Graph ✓" if gv > vv else ("Vector ✓" if vv > gv else "Tie")
            if gv > vv: graph_wins += 1
        else:
            w = "Graph ✓" if gv < vv else ("Vector ✓" if vv < gv else "Tie")
        print(f"  {label:<35} {gv:>8.4f} {vv:>10.4f}  {w:>10}")
        all_metric_rows.append((label, round(gv,4), round(vv,4), w))
 
    print(f"\n  {'Metric':<35} {'Graph RAG':>8} {'Vector RAG':>10}  {'Winner':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10}  {'-'*10}")
    print("  --- RANKING METRICS ---")
    for k in ranking_keys:
        print_row(k, graph_rank_avg[k], vector_rank_avg[k])
    print("  --- LLM-AS-JUDGE: ANSWER QUALITY ---")
    for k in ["relevance","faithfulness","completeness"]:
        print_row(f"Judge:{k.capitalize()}[1-5]", graph_judge_avg[k], vector_judge_avg[k])
    print("  --- LLM-AS-JUDGE: CONTEXT QUALITY ---")
    for k in ["sufficiency","precision","structural_richness"]:
        print_row(f"Context:{k.replace('_',' ').title()}[1-5]", graph_ctx_avg[k], vector_ctx_avg[k])
    print_row("Latency(s)", avg([r["graph_ranking_metrics"]["Latency(s)"] for r in all_results]),
              avg([r["vector_ranking_metrics"]["Latency(s)"] for r in all_results]), higher_is_better=False)
 
    total_quality_metrics = len(ranking_keys) + 3 + 3  # ranking + judge + context
    print(f"\n  Graph RAG wins on {graph_wins}/{total_quality_metrics} quality metrics")
 
    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
 
    full_results = {
        "summary": {
            "graph_rag": {
                "ranking": graph_rank_avg,
                "judge_answer": graph_judge_avg,
                "judge_context": graph_ctx_avg,
            },
            "vector_rag": {
                "ranking": vector_rank_avg,
                "judge_answer": vector_judge_avg,
                "judge_context": vector_ctx_avg,
            },
        },
        "per_query": all_results,
    }
    json_path = os.path.join(out_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Full results → {json_path}")
 
    csv_path = os.path.join(out_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Graph_RAG", "Vector_RAG", "Winner"])
        for row in all_metric_rows:
            w.writerow(row)
    print(f"  Summary CSV → {csv_path}\n")
 
 
if __name__ == "__main__":
    run_comparison()