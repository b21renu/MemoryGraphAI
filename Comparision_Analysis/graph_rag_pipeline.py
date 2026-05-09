"""
graph_rag_pipeline.py
---------------------
Hybrid Graph RAG pipeline.

Retrieval process (the hybrid that is MemoryGraph AI's core claim):
  Step 1 — VECTOR SEARCH: embed the query, find top-k semantically similar
            entity nodes using cosine similarity over TF-IDF embeddings.
  Step 2 — GRAPH TRAVERSAL: from each seed node, follow ALL outgoing and
            incoming edges to collect one-hop neighbours.
  Step 3 — RANKED OUTPUT: seeds first (highest semantic relevance),
            then neighbours sorted by connection count to seeds.

Context for LLM:
  The context string contains RELATIONAL TRIPLES:
    (Source Entity) -[RELATION_TYPE]-> (Target Entity): description
  This is fundamentally different from Vector RAG's flat entity list —
  it gives the LLM STRUCTURAL EVIDENCE about how concepts are connected.

Why this outperforms Vector RAG:
  - Queries about "what does X use?" need the entity X's EDGES, not just X itself
  - Multi-hop queries ("how does A relate to C via B?") require traversal
  - The context string carries relationship semantics that pure vector similarity cannot
"""

import numpy as np
from knowledge_graph import get_graph
from embedder import TFIDFEmbedder
 
 
class GraphRAGPipeline:
    def __init__(self, embedder: TFIDFEmbedder, seed_k: int = 5):
        self.graph = get_graph()
        self.embedder = embedder
        self.seed_k = seed_k
 
        # Pre-compute entity embeddings with name boost
        # The entity name tokens get 3x weight so that a query mentioning "MegaRAG"
        # strongly matches the MegaRAG entity even if 'megarag' has low IDF.
        entity_names = list(self.graph["entities"].keys())
        corpus = [f"{n} {self.graph['entities'][n]}" for n in entity_names]
        entity_matrix = embedder.embed_batch(corpus, name_boosts=entity_names)
        self._entity_names = entity_names
        self._entity_matrix = entity_matrix  # shape (N, D)
 
    def _vector_seeds(self, query: str) -> list[tuple[str, float]]:
        """Top-k entities by cosine similarity to the query."""
        q_vec = self.embedder.embed(query)
        scores = self._entity_matrix @ q_vec  # cosine (vectors are normalised)
        top_idx = np.argsort(scores)[::-1][: self.seed_k]
        return [(self._entity_names[i], float(scores[i])) for i in top_idx]
 
    def retrieve(self, query: str) -> tuple[list[str], list[dict]]:
        """
        Returns:
          ranked_entities — HYBRID ranked list:
            Seeds ordered by cosine similarity.
            Neighbours interleaved by (connection_count * 2 + semantic_score),
            so graph-traversal-found relevant entities surface within top-K.
          triples         — list of {source, relation, target} for context building
        """
        seeds_with_scores = self._vector_seeds(query)
        seed_names = [s for s, _ in seeds_with_scores]
        seed_set = set(seed_names)
 
        # Graph traversal: collect all edges touching seeds
        adjacency = self.graph["adjacency"]
        triples = []
        neighbour_counts: dict[str, int] = {}
 
        for seed in seed_names:
            for rel, neighbour in adjacency.get(seed, []):
                triples.append({"source": seed, "relation": rel, "target": neighbour})
                if neighbour not in seed_set:
                    neighbour_counts[neighbour] = neighbour_counts.get(neighbour, 0) + 1
 
        # Score neighbours: connection_count boosts semantic score
        # This ensures highly-connected neighbours rank within top-K even if
        # their raw semantic score to the query is moderate
        q_vec = self.embedder.embed(query)
        neighbour_scores: dict[str, float] = {}
        for n, cnt in neighbour_counts.items():
            n_idx = self._entity_names.index(n) if n in self._entity_names else -1
            if n_idx >= 0:
                sem_score = float(self._entity_matrix[n_idx] @ q_vec)
            else:
                sem_score = 0.0
            # Hybrid score: semantic + graph connectivity bonus
            neighbour_scores[n] = sem_score + (cnt * 0.15)
 
        # Sort neighbours by hybrid score descending
        ranked_neighbours = sorted(
            neighbour_scores.keys(),
            key=lambda n: neighbour_scores[n],
            reverse=True
        )
 
        # Build final ranked list: seeds first, then neighbours
        seen = set(seed_names)
        ranked_entities = list(seed_names)
        for n in ranked_neighbours:
            if n not in seen:
                seen.add(n)
                ranked_entities.append(n)
 
        return ranked_entities, triples
 
    def get_context_string(self, query: str, max_triples: int = 12) -> str:
        """
        Builds a RELATIONAL context string filtered to the most query-relevant triples.
 
        Each triple is scored by the sum of the semantic similarities of its source
        and target entities to the query. We keep only the top `max_triples` triples.
        This prevents the LLM from receiving noisy/irrelevant relationship chains
        that cause hallucination and looping.
 
        Format: (EntityA) --[relation type]--> (EntityB)
        """
        _, triples = self.retrieve(query)
        if not triples:
            return ""
 
        # Score each triple: how relevant are its two endpoints to the query?
        q_vec = self.embedder.embed(query)
        entity_idx = {name: i for i, name in enumerate(self._entity_names)}
 
        def entity_score(name: str) -> float:
            idx = entity_idx.get(name, -1)
            if idx < 0:
                return 0.0
            return float(self._entity_matrix[idx] @ q_vec)
 
        # Build canonical edge set from the original directed relationships
        canonical_edges = set(
            (src, rel, tgt) for src, rel, tgt in self.graph["relationships"]
        )
 
        # Deduplicate: keep only canonical direction, or the first direction seen
        seen_keys = set()
        scored = []
        for t in triples:
            src, rel, tgt = t["source"], t["relation"], t["target"]
            # Use canonical direction if this triple is in the original graph
            if (src, rel, tgt) in canonical_edges:
                key = (src, rel, tgt)
            elif (tgt, rel, src) in canonical_edges:
                # Reverse: this triple came from the undirected expansion, skip it
                # (the canonical direction will appear separately)
                continue
            else:
                key = (src, rel, tgt)
 
            if key in seen_keys:
                continue
            seen_keys.add(key)
            score = entity_score(src) + entity_score(tgt)
            scored.append((score, src, rel, tgt))
 
        # Sort by relevance score descending, keep top max_triples
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_triples]
 
        lines = []
        for _, src, rel, tgt in top:
            rel_readable = rel.replace("_", " ").lower()
            lines.append(f"({src}) --[{rel_readable}]--> ({tgt})")
 
        return "\n".join(lines)
 
    def answer_question(self, question: str, llm_fn) -> str:
        """Generate a concise, grounded answer citing specific graph relationships."""
        context = self.get_context_string(question)
        if not context.strip():
            return "No relevant information found in the knowledge graph."
 
        prompt = (
            "You are given knowledge graph facts — each line is a verified relationship:\n"
            "  (Entity A) --[relationship]--> (Entity B)\n\n"
            "FACTS:\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Write a precise 2-4 sentence answer using ONLY the facts above.\n"
            "- Name the specific entities and relationship types that answer the question.\n"
            "- Do NOT add any information not present in the facts.\n"
            "- Do NOT repeat any sentence.\n"
            "- Start your answer directly, not with 'Based on the facts...'.\n\n"
            "ANSWER:"
        )
        return llm_fn(prompt)