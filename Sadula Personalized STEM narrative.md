# Personalized STEM Narrative

## 1. System Overview
The **Personalized STEM Narrative** system is a research-grade educational prototype designed for Sri Lankan Grade 10 & 11 students. It dynamically bridges the gap between rigid, standardized textbook material (National Institute of Education syllabus) and individual student interests. 

By combining **Machine Learning (ML) Persona Classification**, **Retrieval-Augmented Generation (RAG)**, and **Generative AI**, the system translates dry scientific facts into highly personalized, culturally relatable stories. Crucially, it shifts from a traditional "read and memorize" model to a **"Tell → Show → Apply"** pedagogical framework, ensuring that students learn the hard science *before* seeing it applied in a narrative that resonates with their personal aspirations.

## 2. Core Architecture
- **Frontend / UI**: Built with **Streamlit** for a seamless, interactive web experience.
- **ML Persona Classifier**: A **scikit-learn Random Forest** model trained on a custom dataset of Sri Lankan student profiles. It maps a student's `Interest` and `Aspiration` to a specific narrative `Theme` (e.g., *Sports Adventure*, *Medical Mystery*).
- **RAG Pipeline (Grounding)**: Uses **ChromaDB** to index and retrieve exact passages from local NIE Grade 10 & 11 textbooks. This ensures all generated content is factually grounded and prevents AI hallucination.
- **Generative AI**: Powered by **Gemini 3 Flash Preview**, configured to output structured JSON that strictly separates the technical explanation from the narrative story.

## 3. The "Tell → Show → Apply" Learning Model
Instead of blending science and storytelling—which often leads to "concept dropping" where students miss the actual science—the system enforces a strict separation:
1. **TELL (The Science Breakdown)**: Precise textbook definitions, equations, and rules are presented *first*.
2. **SHOW (The Personalized Narrative)**: A 3-paragraph story where the student *recognizes* the concept happening in a real-world scenario tailored to their interests. The final paragraph acts as a "Bridge," explicitly connecting the story back to the theory.
3. **APPLY (Assessment Handoff)**: The student is directed to apply their knowledge in a dedicated Quiz Module.

## 4. End-to-End User Flow

### Phase 1: Profiling & Topic Selection
1. **Student Onboarding**: The user inputs their `Interest` (e.g., Cricket), `Aspiration` (e.g., Engineer), and selects a `Difficulty Level` (Low/Medium/High).
2. **Diagnostic Query**: The user selects a specific textbook and chapter (e.g., *Grade 10 Science - Newton's Laws*) and types a specific diagnostic question (e.g., "I don't understand how forces work when pushing a wall").
3. **Generation Trigger**: The user clicks **"Generate Personalized Story"**.

### Phase 2: System Processing (Under the Hood)
1. **Classification**: The ML model predicts the optimal story theme based on the user's profile.
2. **Retrieval (RAG)**: The system searches the local ChromaDB vector store, filtering by the selected textbook, to find the most relevant exact passages addressing the diagnostic query.
3. **Generation**: The Gemini LLM receives the RAG context, the ML theme, and the strict formatting prompt, returning a structured JSON containing the science intro, the 3-paragraph story, key definitions/equations, and exam bullets.

### Phase 3: The Learning Experience
1. **Step 1: Understand the Science First**: The UI renders a clean, distraction-free panel showing the textbook definition, why the concept works, key equations, and a real-world note.
2. **Step 2: See It in Action**: The UI displays the personalized Sri Lankan story.
3. **Feedback Loop**: The user rates the helpfulness of the story (e.g., 😍 Very helpful, 😕 Hard to understand), allowing the system to gather engagement metrics.
4. **Study Sheet (Side Panel)**: A persistent side column displays exactly "What to Write in the Exam" (bullet points) and a "RAG Verification" table showing the exact textbook excerpts used to generate the content, proving factual accuracy.

### Phase 4: Endpoint / Handoff
1. **Handoff to Quiz Module**: The user clicks **"🧪 Go to Quiz Module →"**. The system passes the specific `topic` and `grade` as URL parameters to an external application (e.g., `http://localhost:8502?topic=Newton+Third+Law&grade=Grade+10+Part+I`), handing off the user for formal assessment.
2. **Alternative Route**: The user clicks **"🔄 Learn a New Concept"**, which clears the session state and returns them to Phase 1.
