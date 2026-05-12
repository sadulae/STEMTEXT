# System Architecture & Code Explanation

This document is designed to help you confidently explain the codebase to a panel. It breaks down the files used, their purpose, and provides a deep dive into how the core ML and RAG (Retrieval-Augmented Generation) systems work together.

---

## 1. The Core Files and Their Purpose

The system is broken down into modular components. Here is what each file does:

*   **`main.py` (The Frontend / UI)**
    *   **What it does:** This is the Streamlit application. It handles the user interface, captures student profiles (interest, aspiration), allows topic selection, and beautifully renders the final story, study sheets, and RAG verification tables.
    *   **Key Code:** Uses `st.session_state` to remember the generated story, and imports `StoryEngine` to handle the heavy lifting.
*   **`story_engine.py` (The Brain)**
    *   **What it does:** This is the core logic handler. It contains the `StoryEngine` class which is responsible for three things:
        1.  Predicting the story theme using a traditional Machine Learning model (Random Forest).
        2.  Fetching the relevant textbook context from the Vector Database.
        3.  Prompting the Google Gemini LLM to write the personalized story in a strict JSON format.
*   **`ingest.py` (The Data Pipeline - Offline Process)**
    *   **What it does:** You run this script once whenever you have new PDFs. It reads the official NIE science textbooks, cleans the text, splits it into small paragraphs (chunks), calculates their embeddings, and saves them into a local database folder called `science_vector_db`.
*   **`syllabus_data.py`**
    *   **What it does:** A simple helper file containing a Python dictionary (`CHAPTER_MAP`) that maps book names to chapter names. This populates the dropdowns in the UI.
*   **`persona_model.pkl` & `encoder.pkl`**
    *   **What it is:** These are serialized (saved) versions of a scikit-learn Random Forest model and LabelEncoders. They were trained offline (likely via `train_persona.ipynb`) to predict the best story genre (e.g., "Adventure", "Sci-Fi") based on a student's `Interest` and `Aspiration`.

---

## 2. Deep Dive: How the RAG Pipeline Works

If the panel asks, **"How are you making sure the AI isn't hallucinating the science?"**, you explain the **Retrieval-Augmented Generation (RAG)** pipeline.

RAG works in two distinct phases: **Ingestion (Offline)** and **Retrieval & Generation (Runtime)**.

### Phase 1: Ingestion (Code in `ingest.py`)
Before a student even opens the app, we prepare the textbooks.
1.  **Loading & Cleaning:** `PyMuPDFLoader` reads the PDFs. We use custom logic (the `OFFSETS` dictionary) to figure out exactly what chapter a specific page belongs to.
2.  **Chunking:** We use LangChain's `RecursiveCharacterTextSplitter`. We don't feed the whole book to the AI; we split it into `1000` character chunks with `200` characters of overlap (so sentences don't get cut in half).
3.  **Embedding & Storage:** We pass these text chunks to `GoogleGenerativeAIEmbeddings`. This model converts the text into high-dimensional vectors (arrays of floating-point numbers). We save these vectors into **ChromaDB** (a local vector database) at the `science_vector_db` folder.

### Phase 2: Retrieval & Generation (Code in `story_engine.py`)
When a student clicks "Generate Personalized Story" in the UI, here is what happens:

#### A. Retrieval (`get_story_context` method)
```python
results_with_scores = self.vectorstore.similarity_search_with_score(topic, **search_kwargs)
```
1.  **Query Conversion:** The student's topic and diagnostic query (e.g., "Topic: Chemical bonds. Focus: I struggle with ionic bonds") is converted into a vector using the exact same Google Embeddings model.
2.  **Mathematical Search:** ChromaDB compares the query's vector against all the textbook chunk vectors in the database. It calculates the **L2 Distance** (mathematical proximity in high-dimensional space) to find the most relevant chunks.
3.  **The Score:** The `score` returned by ChromaDB represents this distance. Lower distance means higher semantic similarity. We mathematically convert this distance into a **Similarity Percentage**, proving to the panel that the AI fetched highly relevant textbook material, not random internet data.

#### B. Augmented Generation (`generate_chapter` method)
Now that we have the "Truth" (the textbook chunks), we pass it to the LLM to generate the story.
1.  **The Prompt Construction:** In `generate_chapter()`, we use a LangChain `PromptTemplate`. The prompt explicitly instructs the LLM:
    *   *Factual Context:* Here are the exact passages from the textbook.
    *   *Persona:* The student likes `Cricket` and wants to be an `Engineer`. The theme is `Sports Adventure`.
    *   *Structure:* Follow the "Tell → Show → Apply" design.
2.  **Strict Output (JSON format):** We force the LLM to reply ONLY in a strict JSON format. This ensures the output is predictable. It separates the `"science_intro"` (pure textbook facts) from the `"story"` (the narrative), and extracts `"key_equations"` for the study sheet.

By combining the **Strict Prompting** and the **ChromaDB Context Retrieval**, the system is mathematically constrained to base its stories on the verified Sri Lankan syllabus, practically eliminating hallucinations.
