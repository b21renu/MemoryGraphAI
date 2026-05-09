# MemoryGraph AI: A Dynamic, Context-Aware Hybrid Knowledge Graph Framework for Documents with Comparative Vector Database Analysis

**MemoryGraph AI** is an advanced AI system that transcends traditional **Vector-based RAG (Retrieval-Augmented Generation)** by building a **Structured Knowledge Graph** from unstructured documents such as PDFs, research papers, and text notes.

Instead of retrieving isolated text snippets like conventional AI assistants, MemoryGraph AI constructs a **persistent associative memory** using **Neo4j**, allowing the system to reason across connected concepts.

By representing information as **entities (nodes) and relationships (edges)**, the system enables **multi-hop reasoning**, allowing the AI to connect ideas that may appear in different sections of a document or even across multiple documents.

To further validate the effectiveness of this approach, the project includes a **comparative evaluation framework** that measures the performance of **graph-based retrieval against traditional vector similarity search**.

---

## Key Features
- **Automated Knowledge Extraction:** Converts raw text into Entities (Nodes) and Relationships (Edges).
- **GraphRAG Architecture:** Combines the power of Large Language Models (LLMs) with the factual precision of Graph Databases.
- **Semantic Vector Indexing:** Uses mathematical embeddings to find concepts even if the wording is different.
- **Reasoning Engine:** Traverses the graph to answer complex questions that require "connecting the dots."
- **Performance Analytics:** Real-time tracking of ingestion latency, extraction speed, and answer accuracy.
- **Comparative Retrieval Analysis:** Includes a dedicated module that evaluates the effectiveness of **graph-based reasoning versus vector-based retrieval**, highlighting the strengths and limitations of both approaches.

---
## System Overview

MemoryGraph AI mimics how humans organize knowledge — not as isolated pieces of information, but as a **network of interconnected ideas**.

Instead of relying solely on embeddings, the system constructs a **knowledge graph memory layer**, enabling:

- Context-aware information retrieval
- Relationship-driven reasoning
- Explainable AI outputs supported by graph evidence

To validate the effectiveness of this architecture, the system incorporates **evaluation modules that measure retrieval performance across different retrieval paradigms**.


---

## Technical Architecture

The system operates in a 6-phase pipeline:
1. **Ingestion Layer:** Parses PDFs/Docs and performs Recursive Character Splitting to maintain context.
2. **Extraction Layer (Groq/Llama 3):** Uses LLMs and Pydantic schemas to extract structured JSON triples (Subject -> Relation -> Object).
3. **Graph Construction (Neo4j):** Uses Cypher `MERGE` logic to build a non-redundant, interconnected knowledge web.
4. **Embedding Layer:** Generates 384-dimension vectors for every node using `Sentence-Transformers` for semantic search.
5. **Reasoning Engine:** Converts user queries into vector searches, retrieves the local sub-graph, and synthesizes a grounded answer.
6. **UI Layer (Streamlit):** A professional dashboard for document management and interactive discovery.

---
## Comparative Retrieval Analysis

A key component of this project is the **comparative evaluation between graph-based retrieval and vector-based retrieval systems**.

While vector databases rely on **embedding similarity**, graph-based retrieval leverages **explicit relationships between entities**.

To analyze the effectiveness of both methods, the project includes evaluation pipelines that measure how well each approach retrieves relevant knowledge from documents.

The comparison focuses on answering the following research question:

**Does structured graph reasoning improve contextual retrieval compared to pure vector similarity search?**

---

## Evaluation Framework

The evaluation module tests retrieval performance using **both graph-based and vector-based pipelines**.

### Graph-Based Retrieval Evaluation

This pipeline evaluates how effectively the system retrieves relevant entities using **knowledge graph relationships**.

The approach analyzes:

* Connected entities within the graph
* Contextual relationships
* Multi-hop reasoning accuracy

---

## Vector-Based Retrieval Evaluation

This pipeline retrieves concepts using **embedding similarity and cosine distance**.

The system identifies the closest concept to the query and retrieves related concepts based on semantic similarity.

---

## Evaluation Metrics

To ensure a fair comparison, both retrieval approaches are evaluated using standard **Information Retrieval metrics**.

### Precision

Measures the proportion of retrieved entities that are relevant.

### Recall

Measures how many relevant entities were successfully retrieved.

### F1 Score

Balances precision and recall to provide an overall accuracy measure.


### Contextual Accuracy

Assesses whether retrieved entities maintain meaningful relationships within the knowledge graph.

### Query Latency

Tracks the time required to process and return results.

These metrics provide a **quantitative comparison between graph reasoning and vector similarity retrieval**.

---


## Technology Stack
- **LLM:** Groq (Llama-3.1-8b-instant)
- **Database:** Neo4j AuraDB (Cloud)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Frameworks:** LangChain, NetworkX
- **Frontend:** Streamlit
- **Programming Language:** Python 3.10+

---

## Prerequisites

Before running the app, you need:
1. **Neo4j AuraDB:** Create a free instance at [neo4j.com/cloud/aura/](https://neo4j.com/cloud/aura/). Download your `credentials.txt`.
2. **Groq API Key:** Get a free API key at [console.groq.com](https://console.groq.com/).
3. **Python:** Ensure Python 3.10 or higher is installed.

---

## Installation [MemoryGraphAI Platform]

1. **Clone the project folder and navigate into it:**
   ```bash
   cd MemoryGraphAI
   ```

2. **Install the required dependencies:**
   ```bash
   pip install streamlit neo4j sentence-transformers langchain-groq python-dotenv PyPDF2 python-docx tqdm pandas
   ```

3. **Configure Environment Variables:**
   Create a file named `.env` in the root directory and add your credentials:
   ```env
   GROQ_API_KEY=your_groq_key_here
   NEO4J_URI=neo4j+s://your_instance_id.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   ```

---

## How to Run

The entire system is integrated into a single Streamlit dashboard. You do not need to run the individual phase scripts manually.

1. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Using the Interface:**
   - **Step 1 (Sidebar):** Upload your PDF documents.
   - **Step 2 (Sidebar):** Click **"Build & Index Memory Graph"**. 
     - *Note: This will clean the temporary data folder, extract text, build nodes in Neo4j, and generate vector embeddings.*
   - **Step 3 (Main Screen):** Once the "Graph Ready" message appears, type your question in the search bar.
   - **Step 4 (Evaluation):** Review the **AI Answer** side-by-side with the **Graph Evidence** to ensure the answer is grounded in facts.

---

## Installation [Comparision_Analysis]

1. Navigate to the comparison analysis directory:
   ```bash
   cd Comparison_Analysis
   ```
2. Install the required dependencies:
   ```bash
   pip install requests networkx sentence-transformers numpy
   ```
3. Add your Groq API key inside `generate_nodes_edges.py`:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
---

## How to Run

The comparison analysis consists of three sequential steps.

### Step 1: Generate Nodes and Edges

Extract entities and relationships from the document and create the knowledge graph JSON file.
   ```bash
   python generate_nodes_edges.py
   ```

This script reads `doc.txt` and generates `output.json` containing the extracted nodes and edges.


### Step 2: Run Graph-Based Evaluation

Evaluate the retrieval performance using the graph-based approach.
   ```bash
   python hybrid_approach_evaluation_meterics.py
   ```


### Step 3: Run Vector-Based Evaluation

Evaluate the retrieval performance using vector similarity search.
   ```bash
   python vector_pipeline_evaluation_metrics.py
   ```  
---

## Evaluation & Metrics

The app includes a built-in **Performance & Accuracy Dashboard** that tracks:
- **Ingestion Latency:** Time taken to parse documents.
- **Extraction Speed:** Time taken by the LLM to identify relationships.
- **Node Density:** Total count of unique entities stored in your "Memory."
- **Fact Verification:** A dedicated "Evidence" window showing the exact graph paths used to generate each answer.

---

## Project Structure [MemoryGraphAI Platform]
- `app.py`: The main Streamlit interface and orchestration logic.
- `ingestion.py`: PDF parsing and text cleaning.
- `extraction.py`: LLM-based entity and relationship extraction.
- `graph_builder.py`: Neo4j connection and Cypher query execution.
- `graph_embeddings.py`: Vector generation and indexing.
- `query_engine.py`: The GraphRAG reasoning and answering logic.

---
## Project Structure [Comparision_Analysis]
- `generate_nodes_edges.py`: Extracts entities and relationships using an LLM.
- `hybrid_approach_evaluation_meterics.py`: Evaluates the graph-based retrieval approach.
- `vector_pipeline_evaluation_metrics.py`: Evaluates the vector-based retrieval approach.
- `doc.txt`: Input document for extraction.
- `output.json`: Generated nodes and edges in JSON format.
  
---
# Performance Insights

The integrated analytics module measures:

* Document ingestion latency
* LLM extraction speed
* Graph density
* Retrieval accuracy
* Query response time

These metrics provide insights into **system scalability and retrieval efficiency**.

---
# Output Screenshots
<img width="1600" height="664" alt="1" src="https://github.com/user-attachments/assets/3f321601-ca50-45be-95a4-4b1f5161fedb" />
<p align="center"><b><i>MemoryGraph AI User Interface for Document Upload and Querying</i></b></p>
<br>


<img width="1498" height="828" alt="2" src="https://github.com/user-attachments/assets/875bc58a-9515-4bcb-838b-5059b0417871" />
<p align="center"><b><i>Knowledge Graph Insights and Entity Analysis</i></b></p>
<br>

<img width="1201" height="591" alt="3" src="https://github.com/user-attachments/assets/409c3919-c0c7-4a8c-b557-dbc2391c5137" />
<p align="center"><b><i>Graph Evidence Supporting Generated Answers</i></b></p>
<br>


<img width="1600" height="675" alt="4" src="https://github.com/user-attachments/assets/8c95f19e-4625-4ef8-b33d-97a5ed0b36aa" />
<p align="center"><b><i>Neo4j Graph Database Representation of Stored Knowledge</i></b></p>
<br>

---
# Conclusion

MemoryGraph AI demonstrates how combining **knowledge graphs, semantic embeddings, and LLM reasoning** can create a more **context-aware and explainable AI retrieval system**.

By integrating **graph reasoning with vector search and evaluating both approaches**, the system provides insights into how future AI memory architectures can move beyond traditional retrieval systems toward **structured knowledge-based intelligence**.

---

## Important Notes
- **Rate Limits:** This project uses the Groq Free Tier. If you process very large documents, you may hit the "Tokens Per Day" limit. If this happens, wait a few minutes or switch to the `llama-3.1-8b-instant` model for higher limits.
- **Data Privacy:** Documents are processed locally and then stored in your private Neo4j cloud instance. No data is used to train public models.


