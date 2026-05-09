"""
embedder.py
-----------
Self-contained text embedding using TF-IDF weighted bag-of-words vectors.

Why not sentence-transformers here?
  The container has no network access, so we cannot download model weights.
  TF-IDF over a closed vocabulary (the entity names + descriptions) is
  sufficient for this comparison because:
    1. The vocabulary IS the knowledge graph — every relevant term appears
    2. Cosine similarity over TF-IDF reliably separates relevant from irrelevant
       entities for the queries we use
    3. Both Graph RAG and Vector RAG use IDENTICAL embeddings — the only
       difference tested is graph traversal, not embedding quality

If you run this on a machine with internet access, swap in SentenceTransformer
by setting USE_SENTENCE_TRANSFORMERS = True in the config.
"""

import math
import re
import numpy as np
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class TFIDFEmbedder:
    """
    Fits a TF-IDF vocabulary on a corpus, then embeds arbitrary strings
    into fixed-length L2-normalised vectors.
    """

    def __init__(self, corpus: list[str]):
        """
        corpus: list of strings to build vocabulary from.
        The knowledge graph entity names + descriptions form the corpus.
        """
        self._fit(corpus)

    def _fit(self, corpus: list[str]):
        # Build IDF from corpus
        tokenised = [tokenize(doc) for doc in corpus]
        vocab: set[str] = set()
        for toks in tokenised:
            vocab.update(toks)
        self._vocab = sorted(vocab)
        self._word2idx = {w: i for i, w in enumerate(self._vocab)}
        N = len(corpus)

        # Document frequency
        df = Counter()
        for toks in tokenised:
            for w in set(toks):
                df[w] += 1

        self._idf = np.array(
            [math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0 for w in self._vocab],
            dtype=np.float32,
        )
        self._dim = len(self._vocab)

    def embed(self, text: str, name_boost_text: str = "") -> np.ndarray:
        """
        Embed text with optional name boosting.
        If name_boost_text is provided, its tokens are weighted 3x higher —
        this ensures that entity names dominate over description words, so
        a query containing "MegaRAG" will strongly match the MegaRAG entity
        even when 'megarag' appears in many description strings.
        """
        toks = tokenize(text)
        if not toks:
            return np.zeros(self._dim, dtype=np.float32)
        tf = Counter(toks)
        vec = np.zeros(self._dim, dtype=np.float32)
        for w, cnt in tf.items():
            if w in self._word2idx:
                vec[self._word2idx[w]] = cnt * self._idf[self._word2idx[w]]

        # Apply name boost: tokens in the entity name get 3x weight
        if name_boost_text:
            for w in tokenize(name_boost_text):
                if w in self._word2idx:
                    vec[self._word2idx[w]] *= 3.0

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str], name_boosts: list | None = None) -> np.ndarray:
        if name_boosts:
            return np.stack([self.embed(t, nb) for t, nb in zip(texts, name_boosts)])
        return np.stack([self.embed(t) for t in texts])

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # already normalised


def build_embedder(graph: dict) -> TFIDFEmbedder:
    """
    Build a TFIDFEmbedder from the knowledge graph corpus.
    Corpus = entity names + descriptions + relationship sentences.
    """
    corpus = []
    for name, desc in graph["entities"].items():
        corpus.append(f"{name} {desc}")
    for src, rel, tgt in graph["relationships"]:
        corpus.append(f"{src} {rel.replace('_', ' ')} {tgt}")
    return TFIDFEmbedder(corpus)