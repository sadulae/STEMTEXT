import os
import json
import pickle
import warnings
import logging

# Suppress noisy sklearn version warnings at startup
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Maps human-readable book names to ChromaDB filename metadata values
BOOK_FILENAME_MAP = {
    "Grade 10 - Part I":  "science G-10 P-I E",
    "Grade 10 - Part II": "science G-10 P-II E",
    "Grade 11 - Part I":  "science G-11 P-I E",
    "Grade 11 - Part II": "science G-11  P-II E",
}

class StoryEngine:
    def __init__(self):
        print("Initializing Story Engine...")

        # 1. Load Vector DB (The "Truth")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vectorstore = Chroma(
            persist_directory="science_vector_db",
            embedding_function=embeddings
        )

        # 2. Load ML Persona Model & Encoders (The "Vibe")
        try:
            with open('persona_model.pkl', 'rb') as f:
                self.persona_model = pickle.load(f)
            with open('encoder.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            self.has_model = True
            print("Successfully loaded ML Persona models.")
        except FileNotFoundError:
            print("Warning: persona_model.pkl or encoder.pkl not found.")
            self.has_model = False

        # 3. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)

    def get_theme_for_student(self, interest, aspiration):
        """Predicts the story theme using the trained ML model."""
        if not self.has_model:
            return "Adventure"

        try:
            le_interest   = self.encoders['interest']
            le_aspiration = self.encoders['aspiration']
            le_theme      = self.encoders['theme']
            i_enc = le_interest.transform([interest])[0]
            a_enc = le_aspiration.transform([aspiration])[0]

            import pandas as pd
            X_pred = pd.DataFrame([[i_enc, a_enc]],
                                  columns=['Interest_Encoded', 'Aspiration_Encoded'])
            pred  = self.persona_model.predict(X_pred)
            return le_theme.inverse_transform(pred)[0]
        except Exception as e:
            # Graceful degradation — map common interests manually
            fallback = {
                "Cricket": "Sports Adventure",
                "Gaming": "Sci-Fi/Cyberpunk",
                "Music": "Drama/Inspirational",
                "Reading": "Mystery/Historical",
                "Art": "Creative/Fantasy",
                "Nature": "Exploration",
                "Robotics": "Futuristic/Technology",
            }
            return fallback.get(interest, "Adventure")

    def get_story_context(self, topic, book_name=None):
        """Searches the vector DB for the top 3 most relevant snippets.
        Optionally filters by book (grade-level) to avoid cross-grade results.
        """
        print(f"\nSearching textbook database for: '{topic}'...")

        # Build metadata filter if a specific book is selected
        # Fetch up to 8 chunks to allow for dynamic context sizing
        search_kwargs = {"k": 8}
        if book_name and book_name in BOOK_FILENAME_MAP:
            filename = BOOK_FILENAME_MAP[book_name]
            search_kwargs["filter"] = {"filename": filename}

        results_with_scores = self.vectorstore.similarity_search_with_score(topic, **search_kwargs)

        context = ""
        sources = []
        for doc, score in results_with_scores:
            # Convert ChromaDB distance to a rough similarity percentage
            similarity_pct = max(0.0, min(100.0, (1.0 - (score / 2.0)) * 100.0))
            
            # Dynamic RAG Window: Only keep highly relevant chunks (> 55% similarity)
            # We always keep at least 2 chunks even if the score is lower, as a fallback baseline
            if similarity_pct < 55.0 and len(sources) >= 2:
                continue

            meta = doc.metadata
            source_label = f"{meta.get('filename')} | Chapter: {meta.get('chapter')} | Page: {meta.get('page_number')}"
            context += f"\n--- {source_label} ---\n{doc.page_content}\n"
            
            sources.append({
                "filename": meta.get('filename'),
                "chapter": meta.get('chapter'),
                "page": meta.get('page_number'),
                "similarity": f"{similarity_pct:.1f}%",
                "snippet": doc.page_content[:120].replace('\n', ' ') + "..."
            })

        return context, sources

    def generate_question(self, topic):
        """Generates a standalone pre/post concept-check question."""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
Generate a single, simple multiple-choice question to test a Grade 10-11 Sri Lankan student's understanding of: {topic}

Return ONLY raw JSON. No markdown. Format:
{{
    "question": "The question text",
    "options": {{"A": "...", "B": "...", "C": "..."}},
    "correct_answer": "A"
}}
"""
        )
        response = (prompt | self.llm).invoke({"topic": topic})
        try:
            content = response.content
            if isinstance(content, list):
                raw = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            else:
                raw = str(content)
            start = raw.find('{')
            end   = raw.rfind('}')
            return json.loads(raw[start:end+1])
        except Exception:
            return None

    def generate_chapter(self, student_theme, topic, diagnostic_query,
                         interest="General", aspiration="Student",
                         struggle_level="Medium", book_name=None,
                         pre_fetched_context=None, pre_fetched_sources=None):
        """Generates the full story chapter JSON.
        Accepts pre-fetched context so the DB is NOT queried twice.
        """
        # Use pre-fetched context if provided; otherwise fetch now
        if pre_fetched_context:
            syllabus_context = pre_fetched_context
            sources          = pre_fetched_sources or []
        else:
            syllabus_context, sources = self.get_story_context(
                f"Topic: {topic}. Focus: {diagnostic_query}", book_name
            )

        print(f"Generating story for '{topic}' with theme '{student_theme}'...")

        prompt = PromptTemplate(
            input_variables=["theme","topic","diagnostic","context","interest","aspiration","struggle_level"],
            template="""
You are an expert storyteller and science teacher creating a personalized story for a student.
Topic to teach: {topic}
Student's specific focus / struggle: {diagnostic}

Factual context from the national syllabus textbook:
{context}

CRITICAL INSTRUCTIONS:
1. Genre/theme: {theme}.
2. Protagonist: A Sri Lankan student interested in {interest}, aspiring to be a {aspiration}.
3. Setting: Real Sri Lankan locations, everyday foods, familiar cultural touchpoints.
4. SIMPLICITY — struggle level is '{struggle_level}'. Use simple, relatable analogies (water in a pipe, rolling a coconut). The genre is just flavour; the SCIENCE must be crystal clear.
5. NO alien technology, floating stadiums, or complex Sci-Fi jargon.
6. Connect the science directly to {interest}.

FOLLOW THE "TELL → SHOW → APPLY" LEARNING DESIGN. DO NOT MIX THE SECTIONS:

SECTION 1 — "science_intro" field (MANDATORY):
You MUST provide this section. Write a clear, structured explanation of the concept as a science teacher would explain it in class.
Include:
- What the law/concept is, stated precisely as in the textbook
- Why it works (the underlying reason)
- The key equation(s) written cleanly
- One very short real-world example (1 sentence only)
This must be accurate, concise, and understandable. No story, no characters. Just clear science.

SECTION 2 — "story" field:
You MUST write EXACTLY 3 distinct paragraphs, separated by newlines:
- Paragraph 1 (The Hook): Set the scene and introduce the protagonist and the problem/situation.
- Paragraph 2 (The Action): Develop the problem, showing the science concept happening through action.
- Paragraph 3 (The Bridge - MOST IMPORTANT): The protagonist explicitly pauses to connect their situation to the textbook theory. Use less story and more theory here. Explicitly state the facts, numbers, or laws from the textbook that explain what just happened in the story.

Return ONLY valid JSON (no markdown fences). Ensure "science_intro" is a nested object and "story" is a single string with \n\n for paragraphs:
{{
    "science_intro": {{
        "concept_statement": "The exact law/concept stated in textbook language.",
        "explanation": "2-3 sentences explaining WHY it works in simple but accurate terms.",
        "equations": ["F = ma", "a ∝ F"],
        "real_world_note": "One sentence showing this applies in real life."
    }},
    "story": "Paragraph 1 narrative...\\n\\nParagraph 2 narrative...\\n\\nParagraph 3 explicitly explaining the textbook theory...",
    "key_definitions": [
        {{"term": "Exact term from textbook", "definition": "Textbook-style definition"}},
        {{"term": "Another term", "definition": "Its definition"}}
    ],
    "key_equations": [
        {{"label": "What this represents", "equation": "F = ma"}},
        {{"label": "Another relationship", "equation": "a ∝ F"}}
    ],
    "exam_bullets": [
        "Point 1 — written exactly how a student answers in the O/L exam",
        "Point 2",
        "Point 3"
    ],
    "quiz_topic": "A short, precise topic name to pass to the quiz system (e.g. 'Newton Third Law', 'Photosynthesis Light Reactions')"
}}
"""
        )

        response = (prompt | self.llm).invoke({
            "theme": student_theme, "topic": topic,
            "diagnostic": diagnostic_query, "context": syllabus_context,
            "interest": interest, "aspiration": aspiration,
            "struggle_level": struggle_level
        })

        try:
            content = response.content
            if isinstance(content, list):
                raw = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
            else:
                raw = str(content)
            start = raw.find('{')
            end   = raw.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(raw[start:end+1])
                data['sources'] = sources  # Attach source metadata
                return data
            return None
        except Exception as e:
            print(f"JSON parse error: {e}")
            return None
