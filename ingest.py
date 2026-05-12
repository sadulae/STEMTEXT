import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables (e.g. GOOGLE_API_KEY)
load_dotenv()

# We need to offset the absolute PDF page to match the printed page number
# I determined these offsets by looking at the PDFs directly!
OFFSETS = {
    "science G-10 P-I E": 14,
    "science G-10 P-II E": 10,
    "science G-11  P-II E": 10,
    "science G-11 P-I E": 10,
}

# Mapping: Book Name -> List of (start_page, chapter_name)
CHAPTER_MAP = {
    "science G-10 P-I E": [
        (1, "Chemical basis of life"),
        (23, "Motion in a straight line"),
        (52, "Structure of matter"),
        (84, "Newton's laws of motion"),
        (98, "Friction"),
        (110, "Structure and functions of the plant and animal cell"),
        (123, "Quantification of elements and compounds"),
        (139, "Characteristics of organisms"),
        (156, "Resultant force"),
        (168, "Chemical bonds"),
        (189, "Turning effect of a force"),
        (201, "Equilibrium of forces")
    ],
    "science G-10 P-II E": [
        (1, "The world of life"),
        (27, "Continuity of life"),
        (63, "Hydrostatic pressure and its applications"),
        (86, "Changes in matter"),
        (115, "Rate of reactions"),
        (125, "Work, energy and power"),
        (140, "Current electricity"),
        (169, "Inheritance")
    ],
    "science G-11 P-I E": [
        (1, "Living tissues"),
        (20, "Photosynthesis"),
        (30, "Mixtures"),
        (72, "Waves and their applications"),
        (105, "Geometrical Optics"),
        (143, "Biological processes in human body"),
        (190, "Acids, bases and salts"),
        (202, "Heat changes associated with chemical reactions")
    ],
    "science G-11  P-II E": [
        (1, "Heat"),
        (32, "Power and Energy of Electric Appliances"),
        (50, "Electronics"),
        (79, "Electrochemistry"),
        (115, "Electromagnetism and Electromagnetic Induction"),
        (151, "Hydrocarbons and Their Derivatives"),
        (167, "Organizational levels and interactions of biosphere")
    ]
}

def get_chapter(filename, printed_page):
    if filename not in CHAPTER_MAP:
        return "Unknown Chapter"
    
    chapters = CHAPTER_MAP[filename]
    current_chapter = "Preface / TOC / Noise" # For pages before Chapter 1
    
    for start_page, chap_name in chapters:
        if printed_page >= start_page:
            current_chapter = chap_name
        else:
            break
            
    return current_chapter

def main():
    pdf_folder = "."
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    documents = []
    
    # Load all PDFs
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyMuPDFLoader(pdf_file)
        docs = loader.load()
        
        for doc in docs:
            filename = os.path.basename(doc.metadata.get('source', pdf_file))
            filename_without_ext = os.path.splitext(filename)[0]
            
            # Calculate the actual printed page number using our offsets
            offset = OFFSETS.get(filename_without_ext, 0)
            absolute_pdf_page = doc.metadata.get('page', 0)
            printed_page = absolute_pdf_page - offset + 1
            
            # Determine which chapter this page belongs to
            chapter_name = get_chapter(filename_without_ext, printed_page)
            
            # Skip noise (Preface / TOC) if printed_page is less than 1
            if chapter_name == "Preface / TOC / Noise" or printed_page < 1:
                continue
            
            # Clean up headers/footers
            import re
            clean_text = doc.page_content
            # Remove recurring footers/headers
            clean_text = re.sub(r'(?i)For free distribution', '', clean_text)
            clean_text = re.sub(r'(?i)^\s*(Chemistry|Biology|Physics)\s*$', '', clean_text, flags=re.MULTILINE)
            doc.page_content = clean_text
            
            # Update metadata
            doc.metadata['filename'] = filename_without_ext
            doc.metadata['page_number'] = printed_page
            doc.metadata['chapter'] = chapter_name
            
            documents.append(doc)

    print(f"Total useful pages loaded (after dropping noise): {len(documents)}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    if len(chunks) > 0:
        print(f"Sample metadata from first chunk: {chunks[0].metadata}")

    # Initialize Embeddings
    print("Initializing GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Create Chroma vector store
    persist_directory = "science_vector_db"
    
    print(f"Creating vector database and saving to {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("Done! Vector database successfully created with Chapter-Aware logic.")

if __name__ == "__main__":
    main()
