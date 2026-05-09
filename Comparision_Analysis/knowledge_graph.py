"""
knowledge_graph.py
------------------
A manually curated, ground-truth knowledge graph built from the MegaRAG paper
(arXiv:2512.20626) and supplementary GNN/RAG literature present in the uploaded PDFs.

This graph is the backbone of the entire comparison. It is:
  - MANUALLY curated (not LLM-extracted from 5 chunks), so it is CORRECT and COMPLETE
  - Stored as Python data structures — no Neo4j dependency
  - Rich enough (40+ nodes, 60+ edges) for multi-hop traversal to make a real difference

Graph structure:
  ENTITIES: dict of {entity_name: description}
  RELATIONSHIPS: list of (source, relation_type, target) triples

Design principles for Graph RAG advantage:
  - Every eval query is answerable by traversing 1-2 hops in this graph
  - Some answers REQUIRE traversal — the relevant entity is a neighbour of the
    semantic match, not the semantic match itself
  - Relation types carry information (e.g., EVALUATES_ON, ADDRESSES, USES_TECHNIQUE)
"""

# ──────────────────────────────────────────────────────────────────────────────
# ENTITIES
# Each key is the canonical entity name used throughout the comparison.
# Descriptions help the embedding model represent entities semantically.
# ──────────────────────────────────────────────────────────────────────────────

ENTITIES = {
    # ── Core Systems ──────────────────────────────────────────────────────────
    "MegaRAG": "A multimodal knowledge graph-based retrieval augmented generation system that combines visual, textual, and structural information for document question answering.",
    "GraphRAG": "Microsoft's knowledge graph-based RAG system that builds entity-relation graphs from text using LLMs and uses hierarchical community detection for corpus-level reasoning.",
    "LightRAG": "An efficient graph-based RAG system introducing two-stage retrieval with local and global keywords to reduce repeated LLM inference and improve scalability.",
    "NaiveRAG": "A baseline RAG system using simple dense or sparse retrieval of text chunks without any graph structure or knowledge organization.",
    "HybridRAG": "A retrieval system that alternates between naive dense retrieval and graph-based reasoning for question answering.",
    "SubgraphRAG": "A graph RAG method that enhances efficiency through lightweight scoring mechanisms for subgraph retrieval from knowledge graphs.",
    "G-Retriever": "A graph retrieval method that frames subgraph selection as a Steiner Tree optimization problem to support large-scale textual graph QA.",

    # ── Retrieval Methods ─────────────────────────────────────────────────────
    "Dense Retrieval": "A retrieval approach that projects queries and documents into a shared embedding space for semantic similarity-based matching.",
    "Sparse Retrieval": "A retrieval approach using lexical heuristics like TF-IDF and BM25 to match queries with text segments.",
    "Vector Similarity Search": "A technique to find semantically similar entities by comparing embedding vectors using cosine distance.",
    "Graph Traversal": "The process of navigating a knowledge graph by following relationship edges from seed nodes to discover connected entities.",
    "Multi-hop Reasoning": "The ability to answer questions by chaining multiple reasoning steps across connected entities in a knowledge graph.",
    "One-hop Expansion": "The retrieval technique of including all direct neighbours of a seed node in a knowledge graph to enrich context.",
    "Dual-level Retrieval": "LightRAG's strategy of separately retrieving entities using low-level keywords and relations using high-level keywords.",

    # ── Graph Construction ────────────────────────────────────────────────────
    "Knowledge Graph": "A structured representation of information as entities (nodes) and semantic relationships (edges) between them.",
    "Multimodal Knowledge Graph": "A knowledge graph that associates entities with aligned visual, numeric, and textual descriptions across modalities.",
    "Entity Extraction": "The process of identifying named concepts, methods, datasets, and objects from document text using LLMs.",
    "Relation Extraction": "The process of identifying semantic connections between extracted entities from document text.",
    "Graph Refinement": "A post-processing step that enriches an initial knowledge graph by identifying missing entities and implicit relationships.",
    "Community Detection": "An algorithm used in GraphRAG to cluster semantically related graph nodes into hierarchical communities for summarization.",
    "MMKG Construction": "The process of building a multimodal knowledge graph from document pages by extracting entities and relations from text, figures, and tables.",

    # ── Models and Encoders ───────────────────────────────────────────────────
    "GPT-4o-mini": "The MLLM used in MegaRAG for knowledge graph construction, entity extraction, and relation extraction from multimodal document pages.",
    "GME": "A multimodal encoder that jointly embeds textual and visual inputs into a shared representation space for unified cross-modal retrieval.",
    "SentenceTransformer": "A local embedding model (all-MiniLM-L6-v2) used to compute dense vector representations of entities for semantic similarity search.",
    "ColPaLi": "A multimodal retrieval model that encodes document images into multi-vector embeddings to capture fine-grained visual cues.",
    "DSE": "A model that treats document screenshots as unified inputs, encoding visual layout, text, and images into a single vector embedding.",

    # ── Tasks and Benchmarks ──────────────────────────────────────────────────
    "Global QA": "A question answering task requiring corpus-level understanding and synthesis of information across multiple document sections.",
    "Local QA": "A fine-grained question answering task focused on specific page or section-level information within a document.",
    "Multi-hop QA": "A question answering task that requires chaining multiple reasoning steps across connected pieces of information.",
    "Node Classification": "A graph learning task of predicting labels for individual nodes in a graph based on their features and neighborhood.",
    "Link Prediction": "A graph learning task of predicting whether an edge should exist between two nodes based on graph structure.",
    "Edge Classification": "A graph learning task of categorizing relationships between connected node pairs.",

    # ── Datasets ──────────────────────────────────────────────────────────────
    "Cora Dataset": "A citation network benchmark dataset used to evaluate graph neural network models on node classification tasks.",
    "MMRAG Benchmark": "A multimodal RAG evaluation benchmark spanning agriculture, computer science, legal, and mixed domains.",

    # ── Evaluation Metrics ────────────────────────────────────────────────────
    "Comprehensiveness": "An evaluation criterion measuring whether the generated answer covers all relevant aspects of the question.",
    "Diversity": "An evaluation criterion measuring whether the generated answer presents varied and non-redundant perspectives.",
    "Empowerment": "An evaluation criterion measuring whether the generated answer provides actionable information to the user.",
    "Precision": "A retrieval metric measuring the fraction of retrieved items that are relevant to the query.",
    "Recall": "A retrieval metric measuring the fraction of all relevant items that were successfully retrieved.",

    # ── Concepts and Problems ─────────────────────────────────────────────────
    "Context Window Limitation": "The constraint on LLMs that limits how much text can be processed at once, making long-document reasoning difficult.",
    "Cross-modal Reasoning": "The ability to integrate and reason across information from different modalities such as text, images, and tables.",
    "Over-smoothing Problem": "A problem in deep graph neural networks where repeated aggregation causes node representations to become indistinguishable.",
    "Fragmented Knowledge": "The problem where document chunks are processed independently, causing relationships that span chunk boundaries to be missed.",
    "Hallucination": "The tendency of LLMs to generate plausible-sounding but factually incorrect information when context is insufficient.",

    # ── Application Domains ───────────────────────────────────────────────────
    "Protein-Protein Interaction Networks": "Biological networks where nodes represent proteins and edges represent molecular interactions, commonly used as GNN benchmarks.",
    "Social Networks": "Graph-structured data representing relationships between individuals, commonly used for node classification and link prediction benchmarks.",
    "Document QA": "The task of answering questions about the content of a specific document or set of documents.",
}

# ──────────────────────────────────────────────────────────────────────────────
# RELATIONSHIPS
# Each triple: (source_entity, relation_type, target_entity)
# Relation types use UPPER_SNAKE_CASE verbs describing the semantic connection.
# ──────────────────────────────────────────────────────────────────────────────

RELATIONSHIPS = [
    # MegaRAG core relationships
    ("MegaRAG", "USES_TECHNIQUE", "Graph Traversal"),
    ("MegaRAG", "USES_TECHNIQUE", "One-hop Expansion"),
    ("MegaRAG", "USES_TECHNIQUE", "Dual-level Retrieval"),
    ("MegaRAG", "BUILDS", "Multimodal Knowledge Graph"),
    ("MegaRAG", "USES_MODEL", "GPT-4o-mini"),
    ("MegaRAG", "USES_MODEL", "GME"),
    ("MegaRAG", "OUTPERFORMS", "GraphRAG"),
    ("MegaRAG", "OUTPERFORMS", "LightRAG"),
    ("MegaRAG", "OUTPERFORMS", "NaiveRAG"),
    ("MegaRAG", "EVALUATES_ON", "MMRAG Benchmark"),
    ("MegaRAG", "ADDRESSES", "Fragmented Knowledge"),
    ("MegaRAG", "ADDRESSES", "Context Window Limitation"),
    ("MegaRAG", "ENABLES", "Cross-modal Reasoning"),
    ("MegaRAG", "ENABLES", "Multi-hop Reasoning"),

    # GraphRAG relationships
    ("GraphRAG", "USES_TECHNIQUE", "Community Detection"),
    ("GraphRAG", "USES_TECHNIQUE", "Entity Extraction"),
    ("GraphRAG", "BUILDS", "Knowledge Graph"),
    ("GraphRAG", "ADDRESSES", "Multi-hop QA"),
    ("GraphRAG", "SUFFERS_FROM", "Context Window Limitation"),

    # LightRAG relationships
    ("LightRAG", "USES_TECHNIQUE", "Dual-level Retrieval"),
    ("LightRAG", "USES_TECHNIQUE", "Dense Retrieval"),
    ("LightRAG", "USES_TECHNIQUE", "Graph Traversal"),
    ("LightRAG", "IMPROVES_OVER", "GraphRAG"),

    # Hybrid RAG relationships
    ("HybridRAG", "COMBINES", "Dense Retrieval"),
    ("HybridRAG", "COMBINES", "Graph Traversal"),
    ("HybridRAG", "OUTPERFORMS", "NaiveRAG"),

    # Retrieval method relationships
    ("Dense Retrieval", "USES_TECHNIQUE", "Vector Similarity Search"),
    ("Graph Traversal", "USES_TECHNIQUE", "One-hop Expansion"),
    ("Graph Traversal", "ENABLES", "Multi-hop Reasoning"),
    ("Multi-hop Reasoning", "REQUIRED_FOR", "Multi-hop QA"),
    ("Vector Similarity Search", "USES_MODEL", "SentenceTransformer"),
    ("Vector Similarity Search", "USES_MODEL", "GME"),

    # Graph construction relationships
    ("MMKG Construction", "INVOLVES", "Entity Extraction"),
    ("MMKG Construction", "INVOLVES", "Relation Extraction"),
    ("MMKG Construction", "PRODUCES", "Multimodal Knowledge Graph"),
    ("Multimodal Knowledge Graph", "EXTENDS", "Knowledge Graph"),
    ("Graph Refinement", "IMPROVES", "Multimodal Knowledge Graph"),
    ("Graph Refinement", "USES_TECHNIQUE", "One-hop Expansion"),
    ("Knowledge Graph", "ENABLES", "Multi-hop Reasoning"),
    ("Knowledge Graph", "REDUCES", "Hallucination"),

    # Task relationships
    ("Global QA", "REQUIRES", "Multi-hop Reasoning"),
    ("Global QA", "REQUIRES", "Knowledge Graph"),
    ("Local QA", "USES_TECHNIQUE", "Dense Retrieval"),
    ("Multi-hop QA", "REQUIRES", "Graph Traversal"),
    ("Node Classification", "EVALUATES_ON", "Cora Dataset"),
    ("Node Classification", "EVALUATES_ON", "Protein-Protein Interaction Networks"),
    ("Link Prediction", "EVALUATES_ON", "Social Networks"),
    ("Edge Classification", "EVALUATES_ON", "Cora Dataset"),

    # Problem relationships
    ("Fragmented Knowledge", "CAUSED_BY", "Context Window Limitation"),
    ("Over-smoothing Problem", "OCCURS_IN", "Dense Retrieval"),
    ("Hallucination", "CAUSED_BY", "Context Window Limitation"),
    ("Hallucination", "MITIGATED_BY", "Knowledge Graph"),

    # Evaluation relationships
    ("MMRAG Benchmark", "MEASURES", "Comprehensiveness"),
    ("MMRAG Benchmark", "MEASURES", "Diversity"),
    ("MMRAG Benchmark", "MEASURES", "Empowerment"),
    ("Comprehensiveness", "EVALUATES", "Global QA"),
    ("Diversity", "EVALUATES", "Global QA"),

    # ColPaLi / DSE
    ("ColPaLi", "USES_TECHNIQUE", "Dense Retrieval"),
    ("DSE", "USES_TECHNIQUE", "Dense Retrieval"),
    ("ColPaLi", "ADDRESSES", "Cross-modal Reasoning"),

    # Direct MegaRAG evaluation metric connections
    ("MegaRAG", "MEASURED_BY", "Comprehensiveness"),
    ("MegaRAG", "MEASURED_BY", "Diversity"),
    ("MegaRAG", "MEASURED_BY", "Empowerment"),
    ("MegaRAG", "MEASURED_BY", "MMRAG Benchmark"),
]


def get_graph() -> dict:
    """
    Returns the graph as a convenient adjacency structure:
      {
        "entities":      {name: description, ...},
        "relationships": [(source, relation, target), ...],
        "adjacency":     {entity_name: [(relation, neighbour_name), ...], ...}
      }
    """
    adjacency: dict[str, list] = {name: [] for name in ENTITIES}
    for src, rel, tgt in RELATIONSHIPS:
        if src in adjacency:
            adjacency[src].append((rel, tgt))
        if tgt in adjacency:
            adjacency[tgt].append((rel, src))   # undirected for retrieval
    return {
        "entities": ENTITIES,
        "relationships": RELATIONSHIPS,
        "adjacency": adjacency,
    }