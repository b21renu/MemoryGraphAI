import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Cloud-ready imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# 1. Define the Structure
class Relationship(BaseModel):
    source: str = Field(description="The subject entity (e.g., Graph Neural Networks)")
    relation: str = Field(description="The verb/relationship (e.g., EVALUATED_ON)")
    target: str = Field(description="The object entity (e.g., Cora Dataset)")

class KnowledgeGraph(BaseModel):
    entities: List[str] = Field(description="List of unique entities extracted")
    relationships: List[Relationship] = Field(description="List of directed relationships")

class InformationExtractor:
    def __init__(self):
        # GROQ is perfect for cloud deployment (fast & free tier)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
            
        # self.llm = ChatGroq(
        #     temperature=0,
        #     model_name="llama-3.3-70b-versatile", # powerful model
        #     groq_api_key=api_key
        # )

        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant", # <--- Use this instead
            groq_api_key=api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
        self.prompt = PromptTemplate(
            template="""You are a world-class research assistant. Extract a Knowledge Graph from the following text.
            Focus on Research Methods, Datasets, Tasks, and Metrics.
            
            {format_instructions}
            
            Text to analyze:
            {text}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def chunk_text(self, text: str, chunk_size=4000):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        return splitter.split_text(text)

    def extract(self, text: str, max_chunks: int = 10) -> Dict:
        chunks = self.chunk_text(text)
        full_graph = {"entities": [], "relationships": []}
        
        num_to_process = min(len(chunks), max_chunks)
        print(f"  - Total chunks: {len(chunks)}. Processing {num_to_process}...")
        
        for i in range(num_to_process):
            try:
                print(f"    - Processing Chunk {i+1}/{num_to_process}...")
                _input = self.prompt.format_prompt(text=chunks[i])
                output = self.llm.invoke(_input.to_string())
                
                # Groq is so fast we usually don't need to worry about timeouts
                structured_data = self.parser.parse(output.content)
                
                full_graph["entities"].extend(structured_data.entities)
                for rel in structured_data.relationships:
                    full_graph["relationships"].append({
                        "source": rel.source,
                        "relation": rel.relation,
                        "target": rel.target
                    })
            except Exception as e:
                print(f"    - Error on chunk {i+1}: {e}")
                
        # Clean up duplicates
        full_graph["entities"] = list(set(full_graph["entities"]))
        return full_graph

# --- TEST SCRIPT ---
if __name__ == "__main__":
    from ingestion import DocumentIngestion
    
    ingestor = DocumentIngestion()
    docs = ingestor.process_folder("./data")
    
    if docs:
        extractor = InformationExtractor()
        # Let's try 5 chunks this time (Groq is much faster than OpenAI/Ollama)
        graph_data = extractor.extract(docs[0]['content'], max_chunks=5)
        
        with open("graph_data.json", "w") as f:
            json.dump(graph_data, f, indent=4)
            
        print("\nExtraction Complete! Data saved to graph_data.json")
        print(f"Extracted {len(graph_data['entities'])} unique entities.")