import os
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class MemoryGraphEmbedder:
    def __init__(self):
        # 1. Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 2. Load Embedding Model (Local & Fast)
        # 'all-MiniLM-L6-v2' is small, fast, and great for English text
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def close(self):
        self.driver.close()

    def generate_and_store_embeddings(self):
        """Fetches all nodes, generates embeddings, and saves them back to Neo4j."""
        with self.driver.session() as session:
            # 1. Get all entities
            print("Fetching nodes from Neo4j...")
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            node_names = [record["name"] for record in result]
            
            if not node_names:
                print("No nodes found in database.")
                return

            print(f"Generating embeddings for {len(node_names)} nodes...")
            embeddings = self.model.encode(node_names)

            # 2. Update each node with its embedding
            print("Storing embeddings in Neo4j...")
            update_query = """
            MATCH (e:Entity {name: $name})
            SET e.embedding = $embedding
            """
            
            for name, embedding in zip(node_names, embeddings):
                # Convert numpy array to list for Neo4j
                session.run(update_query, name=name, embedding=embedding.tolist())

            # 3. Create a Vector Index (Required for Phase 5 Reasoning)
            # This allows Neo4j to perform ultra-fast similarity searches
            print("Creating Vector Index...")
            try:
                # 384 is the dimension size of 'all-MiniLM-L6-v2'
                index_query = """
                CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {indexConfig: {
                  `vector.dimensions`: 384,
                  `vector.similarity_function`: 'cosine'
                }}
                """
                session.run(index_query)
                print("Vector Index 'entity_embeddings' created successfully.")
            except Exception as e:
                print(f"Index creation note: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    embedder = MemoryGraphEmbedder()
    embedder.generate_and_store_embeddings()
    embedder.close()
    print("\nPhase 4 Complete: Nodes are now mathematically indexed!")