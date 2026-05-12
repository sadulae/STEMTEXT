import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from story_engine import StoryEngine
from syllabus_data import CHAPTER_MAP

st.set_page_config(
    page_title="Personalized STEM Narrative Simulator",
    layout="wide",
    page_icon="🧬"
)

TOPIC_IMAGES = {
    "biology":  "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?auto=format&fit=crop&q=80&w=800&h=280",
    "physics":  "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&q=80&w=800&h=280",
    "chemistry":"https://images.unsplash.com/photo-1603126857599-f6e157fa2fe6?auto=format&fit=crop&q=80&w=800&h=280",
    "default":  "https://images.unsplash.com/photo-1516979187457-637abb4f9353?auto=format&fit=crop&q=80&w=800&h=280",
}
BIOLOGY_TOPICS   = {"Chemical basis of life","Characteristics of organisms","Living tissues",
                    "Photosynthesis","Biological processes in human body","The world of life",
                    "Continuity of life","Inheritance","Organizational levels and interactions of biosphere"}
CHEMISTRY_TOPICS = {"Structure of matter","Chemical bonds","Quantification of elements and compounds",
                    "Changes in matter","Rate of reactions","Acids, bases and salts","Mixtures",
                    "Electrochemistry","Hydrocarbons and Their Derivatives",
                    "Heat changes associated with chemical reactions"}

def topic_image(chapter):
    if chapter in BIOLOGY_TOPICS:   return TOPIC_IMAGES["biology"]
    if chapter in CHEMISTRY_TOPICS: return TOPIC_IMAGES["chemistry"]
    return TOPIC_IMAGES["physics"]

# ── Configure this URL to point to your teammate's quiz module ────────────────
QUIZ_MODULE_URL = "http://localhost:8502"   # <-- change to teammate's URL/port

def load_engine():
    return StoryEngine()

engine = load_engine()

def reset_session():
    for key in ['story_data','feedback_given','raw_context','sources','theme']:
        st.session_state.pop(key, None)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🧬 Personalized STEM Narrative Simulator")
st.caption("Hybrid ML Personalization + RAG Grounding + Generative AI — Sri Lanka NIE Grade 10 & 11 Syllabus")
st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Student Profile")
    st.caption("These inputs drive the ML Persona Classifier to predict a story theme.")

    interest = st.selectbox("Interest", ["Cricket","Gaming","Music","Reading","Art","Nature","Robotics","Other"])
    if interest == "Other":
        interest = st.text_input("Specify Interest", "Photography")

    aspiration = st.selectbox("Aspiration / Career Goal",
                               ["Engineer","Doctor","Teacher","Athlete","Artist","Scientist","Other"])
    if aspiration == "Other":
        aspiration = st.text_input("Specify Aspiration", "Entrepreneur")

    struggle_level = st.radio(
        "Science Difficulty Level", ["Low","Medium","High"], index=2,
        help="High = simpler analogies. Low = more technical detail."
    )
    st.markdown("---")

    if st.button("🔄 New Topic"):
        reset_session()
        st.rerun()

    with st.expander("ℹ️ System Architecture"):
        st.markdown("""
| Component | Detail |
|-----------|--------|
| LLM | Gemini 3 Flash Preview |
| Embeddings | Gemini Embedding-001 |
| Vector DB | ChromaDB (local, NIE PDFs) |
| Classifier | Random Forest (scikit-learn) |
| Textbooks | Grade 10 & 11 NIE |
        """)

# ── TOPIC SELECTION ───────────────────────────────────────────────────────────
st.header("📚 Topic Selection")
col_book, col_chap = st.columns(2)
with col_book:
    selected_book = st.selectbox("Textbook", list(CHAPTER_MAP.keys()))
with col_chap:
    selected_chapter = st.selectbox("Chapter / Topic", CHAPTER_MAP[selected_book])

st.caption(f"📘 Sourcing from: **{selected_book}** — Sri Lanka National Institute of Education")

diagnostic_query = st.text_area(
    "💬 What specifically are you struggling with or want to focus on?",
    "Explain the core concept in simple terms.",
    height=85
)

# ── GENERATE ──────────────────────────────────────────────────────────────────
if st.button("✨ Generate Personalized Story", type="primary"):
    reset_session()

    with st.status("🚀 Building your personalized narrative...", expanded=True) as status:
        st.write("🔮 Running ML Persona Classifier...")
        theme = engine.get_theme_for_student(interest=interest, aspiration=aspiration)
        st.session_state['theme'] = theme
        st.write(f"   → Predicted theme: **{theme}**")

        st.write("🔍 Searching NIE textbook database (grade-filtered)...")
        context, sources = engine.get_story_context(
            f"Topic: {selected_chapter}. Focus: {diagnostic_query}",
            book_name=selected_book
        )
        st.session_state['raw_context'] = context
        st.session_state['sources']     = sources
        st.write(f"   → Found **{len(sources)}** relevant passages")

        st.write("✍️ Writing Tell → Show content...")
        story_data = engine.generate_chapter(
            student_theme=theme, topic=selected_chapter,
            diagnostic_query=diagnostic_query,
            interest=interest, aspiration=aspiration,
            struggle_level=struggle_level,
            book_name=selected_book,
            pre_fetched_context=context,
            pre_fetched_sources=sources
        )
        status.update(label="Ready! 🎉", state="complete")

    if story_data:
        st.session_state['story_data'] = story_data
        st.rerun()
    else:
        st.error("Story generation failed. Please try again.")

# ── DISPLAY RESULTS ───────────────────────────────────────────────────────────
if 'story_data' in st.session_state:
    story_data  = st.session_state['story_data']
    theme       = st.session_state.get('theme', 'General')
    topic       = selected_chapter
    sources     = st.session_state.get('sources', [])
    raw_context = st.session_state.get('raw_context', '')
    quiz_topic  = story_data.get('quiz_topic', topic)

    st.markdown("---")
    with st.expander("🛠️ Debug: Raw JSON Output"):
        st.json(story_data)

    # ── MAIN CONTENT ──────────────────────────────────────────────────────────
    main_col, side_col = st.columns([2, 1])

    with main_col:
        st.image(topic_image(topic), caption=f"Topic: {topic}", use_container_width=True)

        # ── STEP 1: TELL — Science explanation FIRST ─────────────────────────
        intro = story_data.get('science_intro', {})
        if intro:
            st.markdown("### 📐 Step 1: Understand the Science First")
            st.caption("Read this before the story — it will make the story click instantly.")

            st.markdown(
                f"<div style='background:#0d1b2a;border:1px solid #4dd0e1;border-radius:8px;"
                f"padding:16px 20px;margin-bottom:8px'>"
                f"<span style='color:#4dd0e1;font-weight:700'>📌 The Concept</span><br><br>"
                f"<span style='color:#fff;font-size:0.97rem'>{intro.get('concept_statement','')}</span>"
                f"</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='background:#0d1b2a;border-left:4px solid #81c784;border-radius:6px;"
                f"padding:12px 16px;margin-bottom:8px'>"
                f"<span style='color:#81c784;font-weight:700;font-size:0.8rem'>💡 WHY IT WORKS</span><br><br>"
                f"<span style='color:#eee;font-size:0.92rem'>{intro.get('explanation','')}</span>"
                f"</div>", unsafe_allow_html=True)

            eqs = intro.get('equations', [])
            if eqs:
                eq_html = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(
                    [f"<code style='color:#f59e0b;font-size:1.05rem'>{e}</code>" for e in eqs])
                st.markdown(
                    f"<div style='background:#0d1b2a;border-left:4px solid #f59e0b;border-radius:6px;"
                    f"padding:12px 16px;margin-bottom:8px'>"
                    f"<span style='color:#f59e0b;font-weight:700;font-size:0.8rem'>🔢 KEY EQUATIONS</span>"
                    f"<br><br>{eq_html}</div>", unsafe_allow_html=True)

            if intro.get('real_world_note'):
                st.markdown(
                    f"<div style='background:#1a1a1a;border-left:4px solid #ce93d8;border-radius:6px;"
                    f"padding:10px 16px;margin-bottom:4px'>"
                    f"<span style='color:#ce93d8;font-weight:700;font-size:0.8rem'>🌍 REAL WORLD</span><br>"
                    f"<span style='color:#ddd;font-size:0.88rem'>{intro.get('real_world_note','')}</span>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown("---")

        # ── STEP 2: SHOW — The story ──────────────────────────────────────────
        st.markdown("### 📖 Step 2: See It in Action")
        st.caption("Now read this story. Spot the science concept playing out in real life!")
        st.write(story_data.get('story', ''))

        st.markdown("---")

        # ── STORY FEEDBACK ────────────────────────────────────────────────────
        st.markdown("### 💬 Was This Story Helpful?")
        st.caption("Your feedback helps improve the storytelling for all students.")

        if not st.session_state.get('feedback_given'):
            fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
            with fb_col1:
                if st.button("😍 Very helpful", use_container_width=True):
                    st.session_state['feedback_given'] = "Very helpful"
                    st.rerun()
            with fb_col2:
                if st.button("🙂 Somewhat helpful", use_container_width=True):
                    st.session_state['feedback_given'] = "Somewhat helpful"
                    st.rerun()
            with fb_col3:
                if st.button("😐 Not helpful", use_container_width=True):
                    st.session_state['feedback_given'] = "Not helpful"
                    st.rerun()
            with fb_col4:
                if st.button("😕 Hard to understand", use_container_width=True):
                    st.session_state['feedback_given'] = "Hard to understand"
                    st.rerun()
        else:
            feedback = st.session_state['feedback_given']
            if "Very" in feedback:
                st.success(f"Thanks for your feedback: **{feedback}** 🎉")
            elif "Somewhat" in feedback:
                st.info(f"Thanks for your feedback: **{feedback}**. Try adjusting the difficulty level!")
            else:
                st.warning(f"Thanks for your feedback: **{feedback}**. Try selecting 'High' difficulty or rephrasing your focus question!")

        st.markdown("---")

        # ── NEXT STEP: GO TO QUIZ OR NEW CONCEPT ──────────────────────────────
        st.markdown("### 🎯 Next Steps")
        st.info(
            f"You've just learned about **{quiz_topic}** through a personalized story. "
            f"Ready to test your knowledge or explore something new?"
        )
        
        col_quiz, col_next = st.columns(2)
        with col_quiz:
            quiz_url = f"{QUIZ_MODULE_URL}?topic={quiz_topic.replace(' ', '+')}&grade={selected_book.replace(' ', '+')}"
            st.link_button(
                "🧪 Go to Quiz Module →",
                url=quiz_url,
                use_container_width=True,
                type="primary"
            )
            st.caption(f"Tests you on: **{quiz_topic}**")

        with col_next:
            if st.button("🔄 Learn a New Concept", use_container_width=True):
                reset_session()
                st.rerun()
            st.caption("Clears the page so you can pick a new topic.")

    # ── SIDE PANEL: STUDY SHEET ───────────────────────────────────────────────
    with side_col:
        st.subheader("📋 Study Sheet")
        st.caption("Generated from the same NIE textbook content used to write the story.")

        key_equations = story_data.get('key_equations', [])
        if key_equations:
            st.markdown("#### 🔢 Key Equations")
            for eq in key_equations:
                st.markdown(
                    f"<div style='background:#0d1b2a;border-left:3px solid #f59e0b;"
                    f"border-radius:6px;padding:10px 14px;margin-bottom:8px'>"
                    f"<span style='color:#aaa;font-size:0.75rem'>{eq.get('label','')}</span><br>"
                    f"<code style='color:#f59e0b;font-size:1rem'>{eq.get('equation','')}</code></div>",
                    unsafe_allow_html=True)

        key_definitions = story_data.get('key_definitions', [])
        if key_definitions:
            st.markdown("#### 📖 Key Definitions")
            for defn in key_definitions:
                with st.expander(f"**{defn.get('term','')}**"):
                    st.write(defn.get('definition',''))

        exam_bullets = story_data.get('exam_bullets', [])
        if exam_bullets:
            st.markdown("#### ✏️ What to Write in the Exam")
            for i, bullet in enumerate(exam_bullets, 1):
                st.markdown(
                    f"<div style='background:#0f2027;border-radius:6px;"
                    f"padding:8px 12px;margin-bottom:6px;color:#e0f7fa;font-size:0.88rem'>"
                    f"<b style='color:#4dd0e1'>{i}.</b> {bullet}</div>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔍 RAG Verification")
        st.caption("Exact passages from the NIE textbook — proof the AI is not hallucinating.")
        if sources:
            import pandas as pd
            if sources and 'similarity' in sources[0]:
                df = pd.DataFrame(sources)[['filename','chapter','page','similarity','snippet']]
                df.columns = ["Book","Chapter","Page","Similarity","Excerpt"]
            else:
                df = pd.DataFrame(sources)[['filename','chapter','page','snippet']]
                df.columns = ["Book","Chapter","Page","Excerpt"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            with st.expander("Raw context"):
                st.text(raw_context)
