import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class GraphQueryEngine:
    def __init__(self):
        # 1. Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # 2. Load the Embedding Model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Initialize LLM (8b-instant for speed and limits)
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def close(self):
        self.driver.close()

    def search_graph(self, user_query: str, top_k=5):
        """Finds relevant relationships for text-based context."""
        query_embedding = self.model.encode(user_query).tolist()
        search_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[r]-(neighbor)
        RETURN node.name AS source, type(r) AS relationship, neighbor.name AS target
        LIMIT 15
        """
        context_parts = []
        with self.driver.session() as session:
            result = session.run(search_query, k=top_k, embedding=query_embedding)
            for record in result:
                if record['relationship']: 
                    context_parts.append(f"({record['source']}) -[{record['relationship']}]-> ({record['target']})")
                else: 
                    context_parts.append(f"Entity found: {record['source']}")
        return "\n".join(list(set(context_parts)))

    def get_visualization_data(self, user_query: str, top_k=5):
        """Fetches nodes and edges specifically for the visual graph."""
        query_embedding = self.model.encode(user_query).tolist()
        search_query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $embedding)
        YIELD node, score
        MATCH (node)-[r]-(neighbor)
        RETURN node.name AS source, type(r) AS rel, neighbor.name AS target
        LIMIT 25
        """
        nodes = set()
        edges = []
        with self.driver.session() as session:
            result = session.run(search_query, k=top_k, embedding=query_embedding)
            for record in result:
                src, tgt, rel = record['source'], record['target'], record['rel']
                nodes.add(src)
                nodes.add(tgt)
                edges.append({"source": src, "target": tgt, "label": rel})
        return list(nodes), edges

    def get_graph_analytics(self):
        """Runs Graph Data Science metrics to find influential nodes and themes."""
        # Query 1: Hub Entities (Most connected)
        centrality_query = """
        MATCH (e:Entity)
        RETURN e.name AS name, count{(e)--()} AS connections
        ORDER BY connections DESC
        LIMIT 10
        """
        # Query 2: Theme Clusters (Entities sharing common neighbors)
        community_query = """
        MATCH (e1:Entity)-[:RELATED]-(common)-[:RELATED]-(e2:Entity)
        WHERE e1.name < e2.name
        RETURN e1.name + " & " + e2.name AS pair, count(common) AS strength
        ORDER BY strength DESC
        LIMIT 10
        """
        hubs = []
        communities = []
        with self.driver.session() as session:
            hub_result = session.run(centrality_query)
            for record in hub_result:
                hubs.append({"Entity": record["name"], "Connections": record["connections"]})
            comm_result = session.run(community_query)
            for record in comm_result:
                communities.append({"Cluster": record["pair"], "Strength": record["strength"]})
        return hubs, communities

    def answer_question(self, question: str):
        """Generates a direct answer based on the knowledge graph."""
        graph_context = self.search_graph(question)
        if not graph_context.strip():
            return "No specific information found in the memory graph."

        prompt = f"""
        You are a Knowledge Graph Question Answering system.

        Strict Instructions:
        1. Answer the question using ONLY the concepts present in the Graph Context.
        2. Do NOT infer, guess, or assume relationships that are not explicitly present.
        4. If the answer cannot be derived from the Graph Context, reply exactly:
        "The answer is not present in the knowledge graph."
        5. Provide a clear and structured explanation using multiple sentences.
        6. If definitions or descriptions appear in the Graph Context, include them in the answer.
        7. Do NOT say phrases like "we can infer", "it suggests", or "it is likely".
        8. Explain concepts clearly instead of repeating the same relationship.
        - Also please suggest follow up questions 1 or 2 which are relevant not just how this is related and stuff.
        9. Do not mention the relationship the arrow and stuff in the answer , please add some of your knowledge to it and make it structured

        Graph Context:
        {graph_context}

        Question:
        {question}

        Answer based ONLY on the graph context:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
