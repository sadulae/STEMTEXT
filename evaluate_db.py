import random
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Loading database...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory="science_vector_db", embedding_function=embeddings)
    collection = vectorstore._collection

    print("\n=============================================")
    print(" 1. SPOT CHECKING FOR NOISE")
    print("=============================================")
    # We grab all documents from the database to sample a few random ones
    all_data = collection.get()
    documents = all_data['documents']
    metadatas = all_data['metadatas']
    
    total_docs = len(documents)
    print(f"Total chunks in DB: {total_docs}")
    
    # Pick 3 random chunks to see what they look like
    if total_docs > 0:
        sample_indices = random.sample(range(total_docs), min(3, total_docs))
        for i, idx in enumerate(sample_indices):
            print(f"\n--- Random Chunk {i+1} ---")
            print(f"File: {metadatas[idx].get('filename')} | Page: {metadatas[idx].get('page_number')}")
            # Print a snippet of the text
            print(f"Content: {documents[idx][:250]}...")

    print("\n=============================================")
    print(" 2. CHECKING GRADE 10 TOPICS (Metadata Filtering)")
    print("=============================================")
    # To prove we can isolate Grade 10 topics, we run a query and use a "filter"
    # This filter tells ChromaDB to ONLY look at chunks where the filename contains "G-10"
    
    query = "What are the main topics or chapters in this book?"
    print(f"Searching for: '{query}'")
    print("Filtering by: filename must be 'science G-10 P-I E' or 'science G-10 P-II E'")
    
    # We use the metadata extracted earlier to strictly filter the search
    try:
        results = vectorstore.similarity_search(
            query, 
            k=3,
            filter={"filename": {"$in": ["science G-10 P-I E", "science G-10 P-II E"]}}
        )
        
        for i, doc in enumerate(results):
            print(f"\n--- Grade 10 Result {i+1} ---")
            print(f"File: {doc.metadata.get('filename')} | Page: {doc.metadata.get('page_number')}")
            print(f"Content snippet: {doc.page_content[:250]}...")
            
    except Exception as e:
         print(f"Error filtering: {e}")

if __name__ == "__main__":
    main()
