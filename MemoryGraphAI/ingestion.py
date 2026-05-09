import os
import re
import PyPDF2
from docx import Document
from typing import List, Dict
from tqdm import tqdm

class DocumentIngestion:
    def __init__(self):
        # Basic cleaning regex
        self.clean_regex = re.compile(r'\s+')

    def read_pdf(self, file_path: str) -> str:
        """Extract text from PDF files."""
        text = ""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def read_docx(self, file_path: str) -> str:
        """Extract text from DOCX files."""
        try:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""

    def read_txt(self, file_path: str) -> str:
        """Extract text from TXT files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Removes extra whitespaces, newlines, and non-printable characters."""
        # Replace multiple spaces/newlines with a single space
        text = self.clean_regex.sub(' ', text)
        # Remove non-ascii characters (optional, keep if research papers have symbols)
        text = text.encode("ascii", "ignore").decode()
        return text.strip()

    def process_folder(self, folder_path: str) -> List[Dict]:
        """Iterates through a folder and processes all supported documents."""
        processed_data = []
        files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.docx', '.txt'))]
        
        print(f"Found {len(files)} files. Starting ingestion...")
        
        for filename in tqdm(files):
            file_path = os.path.join(folder_path, filename)
            ext = filename.split('.')[-1].lower()
            
            raw_text = ""
            if ext == 'pdf':
                raw_text = self.read_pdf(file_path)
            elif ext == 'docx':
                raw_text = self.read_docx(file_path)
            elif ext == 'txt':
                raw_text = self.read_txt(file_path)
            
            cleaned_text = self.clean_text(raw_text)
            
            if cleaned_text:
                processed_data.append({
                    "filename": filename,
                    "content": cleaned_text,
                    "char_count": len(cleaned_text)
                })
        
        return processed_data

# --- TEST SCRIPT ---
if __name__ == "__main__":
    # 1. Create a dummy data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # 2. Initialize the Ingestion System
    ingestor = DocumentIngestion()
    
    # 3. Path to your documents
    # Put your PDF/DOCX papers in the 'data' folder
    data_folder = "./data" 
    
    documents = ingestor.process_folder(data_folder)
    
    # 4. Show Result
    print(f"\nSuccessfully ingested {len(documents)} documents.")
    if documents:
        print(f"First document snippet: {documents[0]['content'][:200]}...")