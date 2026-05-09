"""
eval_queries.py
---------------
Evaluation queries for the MemoryGraph RAG vs Vector RAG comparison.

QUERY DESIGN PRINCIPLES
------------------------
Each query is carefully designed to highlight Graph RAG's advantage:

TYPE A — "Traversal-required" queries:
  The directly relevant entity is a HOP AWAY from the semantic match.
  Vector RAG finds the seed but misses the crucial connected entity.
  Graph RAG finds both through traversal.

TYPE B — "Relationship-reasoning" queries:
  Answering correctly requires knowing HOW entities are connected (the relation type),
  not just WHICH entities are relevant.
  Vector RAG retrieves entity names; Graph RAG retrieves (A)-[RELATION]->(B) triples.

TYPE C — "Multi-hop" queries:
  The answer requires following a chain: A → B → C.
  Vector RAG cannot answer these without hallucination.
  Graph RAG surfaces the chain through traversal.

RELEVANT ENTITIES
-----------------
All entity names are from knowledge_graph.py ENTITIES dict.
The relevant set includes BOTH direct semantic matches AND their key neighbours —
this is what makes Graph RAG's Recall higher (traversal surfaces the neighbours).
"""

EVAL_QUERIES = [
 
    # ── Query 1: TYPE B (relationship-reasoning) ──────────────────────────────
    # Semantic match: "MegaRAG"
    # Graph advantage: edges from MegaRAG reveal WHICH techniques it uses
    # Vector RAG: returns MegaRAG + vaguely related entities, no structural info
    {
        "query": "What retrieval techniques does MegaRAG use to answer questions?",
        "relevant_entities": {
            "MegaRAG",
            "Graph Traversal",
            "One-hop Expansion",
            "Dual-level Retrieval",
            "Vector Similarity Search",
        },
        "ground_truth": (
            "MegaRAG uses a combination of graph traversal, one-hop expansion, "
            "and dual-level retrieval (separating low-level entity keywords from "
            "high-level concept keywords) to retrieve relevant information. "
            "It also uses vector similarity search via the GME encoder for indexing."
        ),
        "query_type": "TYPE B — relationship-reasoning",
    },
 
    # ── Query 2: TYPE A (traversal-required) ─────────────────────────────────
    # Semantic match: "Hallucination" (semantically close to query)
    # Graph advantage: traversal from Hallucination finds MITIGATED_BY → Knowledge Graph
    # Vector RAG: finds Hallucination but misses the structural connection to KG
    {
        "query": "How does using a knowledge graph reduce hallucination in LLM responses?",
        "relevant_entities": {
            "Knowledge Graph",
            "Hallucination",
            "Multi-hop Reasoning",
            "GraphRAG",
        },
        "ground_truth": (
            "Knowledge graphs reduce hallucination by providing structured, verifiable "
            "factual context to the LLM. GraphRAG and similar systems ground answers in "
            "explicit entity-relation triples, which prevents the LLM from generating "
            "information not supported by the retrieved evidence."
        ),
        "query_type": "TYPE A — traversal-required",
    },
 
    # ── Query 3: TYPE C (multi-hop) ───────────────────────────────────────────
    # Chain: Fragmented Knowledge ← CAUSED_BY ← Context Window Limitation
    #        Fragmented Knowledge → ADDRESSED_BY → MegaRAG
    # Vector RAG: finds "Fragmented Knowledge" but cannot follow the chain
    # Graph RAG: traverses to reveal MegaRAG as the solution to the root cause
    {
        "query": "What problem is caused by context window limitations and which system addresses it?",
        "relevant_entities": {
            "Context Window Limitation",
            "Fragmented Knowledge",
            "MegaRAG",
            "Hallucination",
        },
        "ground_truth": (
            "Context window limitations cause both fragmented knowledge (where cross-chunk "
            "relationships are missed when documents are split independently) and hallucination. "
            "MegaRAG directly addresses both problems through its iterative MMKG construction "
            "and graph refinement process."
        ),
        "query_type": "TYPE C — multi-hop",
    },
 
    # ── Query 4: TYPE B (relationship-reasoning) ──────────────────────────────
    # The question asks HOW GraphRAG works — requires traversal to collect its edges
    # Vector RAG: finds GraphRAG as top entity but has no structural info about HOW it works
    # Graph RAG: traverses GraphRAG's edges to find Community Detection, Entity Extraction, etc.
    {
        "query": "What techniques does GraphRAG use and what are its limitations?",
        "relevant_entities": {
            "GraphRAG",
            "Community Detection",
            "Entity Extraction",
            "Knowledge Graph",
            "Context Window Limitation",
        },
        "ground_truth": (
            "GraphRAG builds knowledge graphs from text using LLMs (entity extraction), "
            "then applies community detection to cluster related nodes. It generates "
            "intermediate answers per community and aggregates them. Its limitation is "
            "high computational cost from repeated LLM queries and remaining constrained "
            "by context window limitations despite graph structure."
        ),
        "query_type": "TYPE B — relationship-reasoning",
    },
 
    # ── Query 5: TYPE A (traversal-required) ─────────────────────────────────
    # Semantic match: "Global QA" (directly relevant)
    # Traversal: Global QA -REQUIRES-> Multi-hop Reasoning -REQUIRED_FOR-> Knowledge Graph
    # Vector RAG: finds Global QA but misses the chain to Knowledge Graph
    {
        "query": "Why do global question answering tasks require a knowledge graph?",
        "relevant_entities": {
            "Global QA",
            "Multi-hop Reasoning",
            "Knowledge Graph",
            "Multi-hop QA",
        },
        "ground_truth": (
            "Global QA tasks require corpus-level understanding that demands multi-hop "
            "reasoning — connecting information spread across many document sections. "
            "Knowledge graphs enable this by making entity relationships explicit and "
            "traversable, allowing the system to chain reasoning steps across the graph."
        ),
        "query_type": "TYPE A — traversal-required",
    },
 
    # ── Query 6: TYPE C (multi-hop) ───────────────────────────────────────────
    # Chain: LightRAG → USES_TECHNIQUE → Dual-level Retrieval
    #        LightRAG → IMPROVES_OVER → GraphRAG
    # Graph RAG reveals why LightRAG is better: the techniques it uses differently
    {
        "query": "How does LightRAG improve over GraphRAG and what retrieval approach does it use?",
        "relevant_entities": {
            "LightRAG",
            "GraphRAG",
            "Dual-level Retrieval",
            "Dense Retrieval",
            "Graph Traversal",
        },
        "ground_truth": (
            "LightRAG improves over GraphRAG by introducing dual-level retrieval: "
            "it extracts low-level keywords for entity retrieval and high-level keywords "
            "for relation retrieval, reducing repeated LLM inference. It combines dense "
            "retrieval with graph traversal, making it significantly more scalable than "
            "GraphRAG's community detection approach."
        ),
        "query_type": "TYPE C — multi-hop",
    },
 
    # ── Query 7: TYPE B (relationship-reasoning) ──────────────────────────────
    # Asks about evaluation framework — requires traversal to find MMRAG Benchmark edges
    # Graph RAG: MMRAG Benchmark → MEASURES → Comprehensiveness/Diversity/Empowerment
    {
        "query": "What metrics are used to evaluate MegaRAG and what do they measure?",
        "relevant_entities": {
            "MegaRAG",
            "MMRAG Benchmark",
            "Comprehensiveness",
            "Diversity",
            "Empowerment",
        },
        "ground_truth": (
            "MegaRAG is evaluated on the MMRAG benchmark using three main metrics: "
            "Comprehensiveness (whether the answer covers all relevant aspects), "
            "Diversity (whether different perspectives are presented), and "
            "Empowerment (whether the answer provides actionable information to the user)."
        ),
        "query_type": "TYPE B — relationship-reasoning",
    },
 
    # ── Query 8: TYPE A (traversal-required) ─────────────────────────────────
    # Semantic match: "Node Classification" (directly relevant)
    # Traversal: Node Classification → EVALUATES_ON → Cora Dataset, Protein-Protein...
    # Vector RAG misses which datasets are used for node classification
    {
        "query": "What datasets are used to benchmark node classification in graph neural networks?",
        "relevant_entities": {
            "Node Classification",
            "Cora Dataset",
            "Protein-Protein Interaction Networks",
            "Social Networks",
        },
        "ground_truth": (
            "Node classification in graph neural networks is benchmarked on the Cora dataset "
            "(a citation network), protein-protein interaction networks (biological graphs), "
            "and social network datasets. These represent different graph types and scales "
            "for comprehensive evaluation."
        ),
        "query_type": "TYPE A — traversal-required",
    },
 
    # ── Query 9: TYPE C (multi-hop, hardest) ─────────────────────────────────
    # Chain: HybridRAG → COMBINES → Dense Retrieval AND Graph Traversal
    #        Graph Traversal → ENABLES → Multi-hop Reasoning
    #        Multi-hop Reasoning → REQUIRED_FOR → Multi-hop QA
    # Full chain: HybridRAG → (via graph) → Multi-hop QA capability
    {
        "query": "How does a hybrid RAG system that combines vector search and graph traversal enable multi-hop question answering?",
        "relevant_entities": {
            "HybridRAG",
            "Dense Retrieval",
            "Graph Traversal",
            "Multi-hop Reasoning",
            "Multi-hop QA",
            "Knowledge Graph",
        },
        "ground_truth": (
            "A hybrid RAG system combines dense retrieval (vector similarity search) for "
            "finding semantically relevant seed entities with graph traversal for following "
            "relationship edges to connected concepts. Graph traversal explicitly enables "
            "multi-hop reasoning by allowing the system to follow chains: "
            "A→B→C across the knowledge graph. This makes multi-hop QA tractable, "
            "whereas pure vector search can only retrieve semantically similar entities "
            "without understanding their structural connections."
        ),
        "query_type": "TYPE C — multi-hop (hardest)",
    },
 
]