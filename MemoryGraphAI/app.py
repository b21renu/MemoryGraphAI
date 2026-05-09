import streamlit as st
import os
import shutil
import time
import pandas as pd

# Import custom modules
from ingestion import DocumentIngestion
from extraction import InformationExtractor
from graph_builder import MemoryGraphBuilder
from graph_embeddings import MemoryGraphEmbedder
from query_engine import GraphQueryEngine
from streamlit_agraph import agraph, Node, Edge, Config

# Page Configuration
st.set_page_config(page_title="MemoryGraph AI", layout="wide", page_icon="🧠")
st.title("MemoryGraph AI: Context-Aware Hybrid Knowledge Graph")

# Initialize session state
if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_nodes" not in st.session_state:
    st.session_state.last_nodes = []

if "last_edges" not in st.session_state:
    st.session_state.last_edges = []


# Load tools
@st.cache_resource
def load_tools():
    return {
        "ingestor": DocumentIngestion(),
        "extractor": InformationExtractor(),
        "query_engine": GraphQueryEngine(),
    }


tools = load_tools()

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("⚙️ System Controls")

    clear_db = st.checkbox("Wipe Neo4j (Fresh Start)")
    uploaded_files = st.file_uploader(
        "Upload PDFs", accept_multiple_files=True, type=["pdf"]
    )

    process_btn = st.button("Build & Index Graph")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []

    if process_btn and uploaded_files:

        overall_start = time.time()

        with st.status("Pipeline Processing...", expanded=True) as status:

            if os.path.exists("data"):
                shutil.rmtree("data")

            os.makedirs("data", exist_ok=True)

            for f in uploaded_files:
                with open(os.path.join("data", f.name), "wb") as b:
                    b.write(f.read())

            # Ingestion
            t0 = time.time()
            docs = tools["ingestor"].process_folder("./data")
            ingest_time = time.time() - t0

            # Extraction
            t1 = time.time()
            builder = MemoryGraphBuilder()

            if clear_db:
                builder.clear_database()

            for doc in docs:

                st.write(f"Extracting: {doc['filename']}...")

                graph_data = tools["extractor"].extract(doc["content"], max_chunks=5)

                builder.build_graph(graph_data)

            builder.close()

            extract_time = time.time() - t1

            # Embeddings
            t2 = time.time()

            embedder = MemoryGraphEmbedder()
            embedder.generate_and_store_embeddings()
            embedder.close()

            embed_time = time.time() - t2

            st.session_state.metrics.append(
                {
                    "Timestamp": time.strftime("%H:%M:%S"),
                    "Files": len(uploaded_files),
                    "Ingest(s)": round(ingest_time, 2),
                    "Extract(s)": round(extract_time, 2),
                    "Embed(s)": round(embed_time, 2),
                    "Total(s)": round(time.time() - overall_start, 2),
                }
            )

            status.update(label="Graph Ready!", state="complete")


# ---------------- MAIN TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Search & Reasoning",
        "📊 Knowledge Discovery",
        "📊 Performance Metrics",
        "📄Graph Evidence",
    ]
)


# ------------------------------------------------
# TAB 1 : CHAT
# ------------------------------------------------
with tab1:

    st.subheader("💬 Chat with MemoryGraph")

    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask your question...")

    if user_query:

        st.session_state.chat_history.append({"role": "user", "content": user_query})

        q_start = time.time()

        with st.spinner("Analyzing Memory Graph..."):

            answer = tools["query_engine"].answer_question(user_query)

            node_names, edge_data = tools["query_engine"].get_visualization_data(
                user_query
            )

        q_latency = time.time() - q_start

        st.session_state.last_nodes = node_names
        st.session_state.last_edges = edge_data

        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"{answer}\n\nLatency: {q_latency:.2f}s"}
        )

        st.rerun()


# ------------------------------------------------
# TAB 2 : KNOWLEDGE DISCOVERY
# ------------------------------------------------
with tab2:

    subtab1, subtab2 = st.tabs(["🧠 Knowledge Concepts", "🌐 Graph View"])

    # -------- Graph Data Science Insights --------
    with subtab1:

        st.header(" Graph Insights")

        st.write("Analyze the global structure of your Knowledge Graph.")

        if st.button("Run Global Analytics"):

            with st.spinner("Calculating Graph Metrics..."):

                hubs, communities = tools["query_engine"].get_graph_analytics()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Hub Entities")
                    st.table(pd.DataFrame(hubs))

                with col2:
                    st.subheader("Concept Clusters")
                    st.table(pd.DataFrame(communities))

    # -------- Graph Visualization --------
    with subtab2:

        st.subheader(" Graph Visualization")

        if st.session_state.last_nodes:

            nodes = [
                Node(id=n, label=n, size=25, color="#007bff")
                for n in st.session_state.last_nodes
            ]

            edges = [
                Edge(source=e["source"], target=e["target"], label=e["label"])
                for e in st.session_state.last_edges
            ]

            config = Config(
                width="100%",
                height=800,
                directed=True,
                physics=True,
                nodeHighlightBehavior=True,
            )

            agraph(nodes=nodes, edges=edges, config=config)

        else:
            st.info("Ask a question in Search & Reasoning tab to generate a graph.")


# ------------------------------------------------
# TAB 3 : PERFORMANCE METRICS
# ------------------------------------------------
with tab3:

    st.header("📊 Performance Metrics")

    if st.session_state.metrics:

        df = pd.DataFrame(st.session_state.metrics)

        st.dataframe(df, use_container_width=True)

    else:
        st.info("No pipeline runs recorded yet.")


# --- TAB 4 : Graph Evidence ---
with tab4:
    st.subheader("📄 Graph Evidence for Last Query")

    if st.session_state.chat_history:
        # Find the last user query
        last_user_query = next(
            (
                m["content"]
                for m in reversed(st.session_state.chat_history)
                if m["role"] == "user"
            ),
            None,
        )

        if last_user_query:
            # Get textual evidence from your query engine
            evidence = tools["query_engine"].search_graph(last_user_query)

            if evidence:
                st.code(evidence)
            else:
                st.info("No evidence found for the last query.")
        else:
            st.info("No user query found in chat history.")
    else:
        st.info(
            "Ask a question in the Search & Reasoning tab to generate graph evidence."
        )
