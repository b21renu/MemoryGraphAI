"""
vector_pipeline.py
------------------
Pure Vector RAG pipeline — cosine similarity only, no graph traversal.

This is the BASELINE. It retrieves the top-k entities semantically similar
to the query and passes them as a FLAT LIST to the LLM — no relationship
information, no structural context.

The comparison is designed to isolate the graph traversal benefit:
  - SAME embedding model (TFIDFEmbedder)
  - SAME entity pool (the same knowledge graph nodes)
  - SAME LLM for answer generation (Groq / Anthropic)
  - DIFFERENT context: flat names vs. relational triples

Any metric advantage for Graph RAG is therefore attributable SOLELY to
the graph traversal and relational context — not embedding quality or model choice.
"""

import numpy as np
from knowledge_graph import get_graph
from embedder import TFIDFEmbedder
 
 
class VectorRAGPipeline:
    def __init__(self, embedder: TFIDFEmbedder, top_k: int = 5):
        self.graph = get_graph()
        self.embedder = embedder
        self.top_k = top_k
 
        entity_names = list(self.graph["entities"].keys())
        corpus = [f"{n} {self.graph['entities'][n]}" for n in entity_names]
        entity_matrix = embedder.embed_batch(corpus, name_boosts=entity_names)
        self._entity_names = entity_names
        self._entity_matrix = entity_matrix
 
    def search(self, query: str) -> tuple[list[str], list[float]]:
        """Returns (entity_names, scores) for top-k entities by cosine similarity."""
        q_vec = self.embedder.embed(query)
        scores = self._entity_matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        names = [self._entity_names[i] for i in top_idx]
        sims = [float(scores[i]) for i in top_idx]
        return names, sims
 
    def get_context_string(self, query: str) -> str:
        """
        Flat entity list — no relationship information.
        The LLM receives only entity names and their descriptions.
        No structural connections are provided.
        """
        names, scores = self.search(query)
        if not names:
            return ""
        lines = []
        for name, score in zip(names, scores):
            desc = self.graph["entities"].get(name, "")
            lines.append(f"Entity: {name} (relevance={score:.3f})\n  Description: {desc}")
        return "\n\n".join(lines)
 
    def answer_question(self, question: str, llm_fn) -> str:
        """Generate a concise answer using the flat entity context."""
        context = self.get_context_string(question)
        if not context.strip():
            return "No relevant information found via vector search."
 
        prompt = (
            "You are given a list of entities retrieved by semantic similarity.\n\n"
            "RETRIEVED ENTITIES:\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Write a precise 2-4 sentence answer using ONLY the entities listed above.\n"
            "- Do NOT add any information not in the entity descriptions.\n"
            "- Do NOT repeat any sentence.\n"
            "- Start your answer directly.\n\n"
            "ANSWER:"
        )
        return llm_fn(prompt)
 