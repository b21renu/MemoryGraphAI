import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class MemoryGraphBuilder:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        # Establish connection to Neo4j AuraDB
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Removes all nodes and relationships (USE WITH CAUTION)."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def build_graph(self, data: dict):
        """Iterates through extracted data and creates graph elements."""
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        with self.driver.session() as session:
            # 1. Create Nodes
            print(f"Creating {len(entities)} nodes...")
            for entity in entities:
                # We use 'Concept' as a generic label. 
                # In Phase 5 we can make this more specific (e.g., Paper, Method).
                session.run(
                    "MERGE (e:Entity {name: $name})",
                    name=entity.strip().title()
                )

            # 2. Create Relationships
            print(f"Creating {len(relationships)} relationships...")
            for rel in relationships:
                # This Cypher query finds two nodes and connects them
                query = """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
                """
                session.run(
                    query,
                    source=rel['source'].strip().title(),
                    target=rel['target'].strip().title(),
                    rel_type=rel['relation'].replace(" ", "_").upper()
                )

# --- EXECUTION ---
if __name__ == "__main__":
    # Load the JSON from Phase 2
    if not os.path.exists("graph_data.json"):
        print("Error: graph_data.json not found. Run extraction.py first.")
    else:
        with open("graph_data.json", "r") as f:
            extracted_data = json.load(f)

        builder = MemoryGraphBuilder()
        
        # Optional: Start with a clean slate
        # builder.clear_database() 
        
        builder.build_graph(extracted_data)
        builder.close()
        
        print("\nGraph Construction Successful!")
        print("You can now view your graph at the Neo4j Aura Console.")