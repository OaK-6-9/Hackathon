import json
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. TEXT EXTRACTION & CHUNKING
# =====================================================================

def extract_pdf_text(uploaded_file) -> str:
    """Extract clean text from an uploaded PDF file."""
    pdf_reader = PdfReader(uploaded_file)
    extracted_text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_text += f"\n[Page {page_num + 1}]\n" + page_text
    return extracted_text.strip()


def extract_docx_text(uploaded_file) -> str:
    """Extract text from paragraphs and tables in a DOCX file."""
    doc = Document(uploaded_file)
    parts = []

    for idx, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()
        if text:
            parts.append(f"[Paragraph {idx}] {text}")

    for table_idx, table in enumerate(doc.tables, start=1):
        parts.append(f"[Table {table_idx}]")
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " | ") for cell in row.cells]
            parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_pptx_text(uploaded_file) -> str:
    """Extract text from slide titles, text boxes, tables, and notes in a PPTX file."""
    prs = Presentation(uploaded_file)
    parts = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        parts.append(f"\n[Slide {slide_num}]")

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text and shape.text.strip():
                parts.append(shape.text.strip())

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [cell.text.strip().replace("\n", " | ") for cell in row.cells]
                    parts.append(" | ".join(cells))

        if slide.has_notes_slide:
            notes = []
            for shape in slide.notes_slide.notes_text_frame.paragraphs:
                text = shape.text.strip()
                if text:
                    notes.append(text)
            if notes:
                parts.append("[Speaker Notes]")
                parts.extend(notes)

    return "\n".join(parts).strip()


def extract_document_text(uploaded_file, file_type: str) -> str:
    """Route the uploaded file to the correct text extractor."""
    if file_type == "pdf":
        return extract_pdf_text(uploaded_file)
    if file_type == "docx":
        return extract_docx_text(uploaded_file)
    if file_type == "pptx":
        return extract_pptx_text(uploaded_file)
    raise ValueError(f"Unsupported file type: {file_type}")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Splits document text into overlapping paragraph chunks for RAG search."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# =====================================================================
# 2. LOCAL RAG & OLLAMA LLM INTEGRATION
# =====================================================================

class IntelligentPDFEngine:
    def __init__(self, ollama_model: str = "llama3.2:1b"):
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        # Load local embedding model (~80MB RAM)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _call_ollama(self, prompt: str, json_format: bool = False) -> str:
        """Sends a request to the local Ollama instance."""
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }
        if json_format:
            payload["format"] = "json"

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return "Error: Unable to connect to local Ollama server."
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running (`ollama serve`)."

    def get_relevant_chunks(self, query: str, chunks: list, top_k: int = 3) -> list:
        """Finds the most relevant passages from the PDF using vector search."""
        if not chunks:
            return []
        query_vec = self.embedder.encode([query])
        chunk_vecs = self.embedder.encode(chunks)
        sims = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in top_indices]

    def generate_summary(self, full_text: str, chunks: list) -> dict:
        """Generate a comprehensive topic-by-topic summary using the entire document.

        Uses hierarchical summarization so long documents are not truncated. Every
        chunk is summarized, those summaries are grouped and merged, and the final
        model call creates a structured study guide.
        """
        if not full_text.strip() or not chunks:
            return {"overview": "No readable content found.", "sections": [], "key_takeaways": []}

        # PASS 1: summarize every chunk. This makes sure content from every page/slide
        # or document section gets represented, rather than using only the first part.
        chunk_summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            prompt = f"""You are an academic tutor. Summarize the following document section in detail.
Cover ALL concepts, definitions, steps, examples, formulas, names, important facts,
subtopics, and relationships that appear in this section. Do not invent information.
Write 5-10 concise but informative bullet points. This is section {idx} of the document.

SECTION TEXT:
{chunk}

DETAILED SECTION SUMMARY:"""
            result = self._call_ollama(prompt, json_format=False)
            if result and not result.startswith("Error:"):
                chunk_summaries.append(f"Section {idx}:\n{result.strip()}")

        if not chunk_summaries:
            return {
                "overview": "The document could not be summarized because the local model did not return usable output.",
                "sections": [],
                "key_takeaways": []
            }

        # PASS 2: merge small groups of chunk summaries. No middle chunks are dropped.
        group_summaries = []
        group_size = 6
        for group_start in range(0, len(chunk_summaries), group_size):
            group = chunk_summaries[group_start:group_start + group_size]
            group_text = "\n\n==========\n\n".join(group)
            prompt = f"""You are building a comprehensive study guide from consecutive sections of a document.
Use ONLY the supplied section summaries. Combine related concepts but keep every distinct topic.
Preserve important definitions, steps, formulas, examples, terminology, and facts.
Do not invent or add outside information.

Return a detailed topic outline in plain text. Use headings and bullet points.

SECTION SUMMARIES:
{group_text}

GROUPED TOPIC SUMMARY:"""
            result = self._call_ollama(prompt, json_format=False)
            if result and not result.startswith("Error:"):
                group_summaries.append(f"Group {group_start // group_size + 1}:\n{result.strip()}")

        if not group_summaries:
            group_summaries = chunk_summaries

        # PASS 3: final merge into structured JSON. Since this prompt contains only
        # grouped summaries, it remains manageable even for large documents while
        # retaining coverage of all source chunks.
        grouped_text = "\n\n==============================\n\n".join(group_summaries)
        prompt = f"""You are an expert academic tutor creating a comprehensive study guide from a complete document.
Use ONLY the supplied grouped summaries. Do not add outside facts.

Create a LONG, detailed summary that covers EVERY topic represented in the material.
Group related content into logical topics, but do not combine unrelated topics just to make the answer shorter.
The final summary should be useful for exam preparation and should preserve the document's terminology.

For each topic include:
- Topic title
- Detailed explanation
- Important subtopics/concepts
- Definitions, formulas, steps, examples, or facts mentioned in the source when present
- Important points students should remember

Also provide:
- A 4-6 sentence overall overview of the complete document
- 8-15 important final key takeaways

Return ONLY valid JSON matching this exact structure:
{{
  "overview": "Detailed overall overview of the complete document.",
  "sections": [
    {{
      "title": "Topic name",
      "summary": "Detailed explanation of this topic.",
      "key_points": ["Point 1", "Point 2", "Point 3", "Point 4"]
    }}
  ],
  "key_takeaways": [
    "Important takeaway 1",
    "Important takeaway 2"
  ]
}}

GROUPED SUMMARIES:
{grouped_text}
"""

        response_text = self._call_ollama(prompt, json_format=True)
        try:
            data = json.loads(response_text)
            data.setdefault("overview", "")
            data.setdefault("sections", [])
            data.setdefault("key_takeaways", [])
            return data
        except Exception:
            # Fallback: display all grouped summaries rather than losing coverage.
            return {
                "overview": "A comprehensive section-by-section summary is shown below because the final structured JSON could not be parsed.",
                "sections": [
                    {
                        "title": f"Topic Group {idx + 1}",
                        "summary": text.split(":\n", 1)[-1],
                        "key_points": []
                    }
                    for idx, text in enumerate(group_summaries)
                ],
                "key_takeaways": [
                    "The summary above was generated from all readable sections of the uploaded document."
                ]
            }

    def generate_quiz(self, chunks: list, num_questions: int = 3) -> list:
        """Generates real multiple-choice questions from document passages."""
        selected_text = "\n\n".join(chunks[:3]) # Use initial context chunks
        
        prompt = f"""You are an exam writer. Based strictly on the text provided below, generate {num_questions} multiple-choice questions.

Source Text:
{selected_text}

Respond ONLY in valid JSON matching this structure:
{{
  "questions": [
    {{
      "id": 1,
      "question": "Clear question based on the text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_index": 0,
      "explanation": "Brief explanation referencing the source text."
    }}
  ]
}}"""

        response_text = self._call_ollama(prompt, json_format=True)
        try:
            data = json.loads(response_text)
            return data.get("questions", [])
        except Exception:
            return []

    def answer_question(self, user_question: str, context_chunks: list) -> str:
        """Answers user queries grounded strictly in retrieved PDF context."""
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""You are a helpful AI study assistant. Answer the user's question using ONLY the context provided from their document. If the answer cannot be found in the context, state "I couldn't find that specific information in the document."

Document Passages:
{context_str}

User Question: {user_question}
Answer:"""

        return self._call_ollama(prompt, json_format=False)

# =====================================================================
# 3. STREAMLIT INTERFACE
# =====================================================================

def main():
    st.set_page_config(page_title="PDF AI Learning Assistant", page_icon="📄", layout="wide")
    
    # Initialize session state
    if "engine" not in st.session_state:
        st.session_state.engine = IntelligentPDFEngine()
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar Config
    st.sidebar.title("⚙️ Local Model Settings")
    model_name = st.sidebar.text_input("Ollama Model Name", value="llama3.2:1b")
    if model_name != st.session_state.engine.ollama_model:
        st.session_state.engine.ollama_model = model_name

    st.sidebar.markdown("---")
    st.sidebar.title("📄 Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or PPTX",
        type=["pdf", "docx", "pptx"]
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if st.sidebar.button("Process Document 🚀", type="primary"):
            with st.spinner("Extracting text and indexing document..."):
                text = extract_document_text(uploaded_file, file_type)
                if not text:
                    st.sidebar.error("No readable text was found in this file.")
                else:
                    chunks = chunk_text(text)
                    st.session_state.pdf_text = text
                    st.session_state.pdf_chunks = chunks
                    st.session_state.file_type = file_type
                    st.session_state.file_name = uploaded_file.name
                    # Clear cached previous document data
                    st.session_state.pop("summary_data", None)
                    st.session_state.pop("quiz_data", None)
                    st.session_state.chat_history = []
                    st.sidebar.success(f"Indexed {len(chunks)} chunks from {file_type.upper()}!")

    # Main Content Area
    st.title("📚 Intelligent Document Learning Assistant")

    if not st.session_state.pdf_text:
        st.info("👈 Upload a PDF, DOCX, or PPTX file from the sidebar and click **Process Document** to begin.")
        return

    tab1, tab2, tab3 = st.tabs(["📖 Document Summary", "❓ Practice Quiz", "💬 Document Q&A"])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.header("📖 AI Document Summary")
        if st.button("Generate Detailed Summary ✨", type="primary"):
            with st.spinner("Reading the complete document and building a topic-by-topic summary..."):
                summary = st.session_state.engine.generate_summary(
                    st.session_state.pdf_text,
                    st.session_state.pdf_chunks
                )
                st.session_state.summary_data = summary

        if "summary_data" in st.session_state:
            s = st.session_state.summary_data
            st.subheader("📌 Complete Overview")
            st.write(s.get("overview", "No overview generated."))

            sections = s.get("sections", [])
            if sections:
                st.subheader("📚 Detailed Topic-by-Topic Summary")
                for idx, section in enumerate(sections, start=1):
                    with st.container(border=True):
                        st.markdown(f"### {idx}. {section.get('title', 'Topic')}" )
                        st.write(section.get("summary", ""))
                        points = section.get("key_points", [])
                        if points:
                            st.markdown("**Important points:**")
                            for point in points:
                                st.markdown(f"- {point}")

            st.subheader("🎯 Key Takeaways")
            takeaways = s.get("key_takeaways", [])
            for pt in takeaways:
                st.markdown(f"- {pt}")

    # --- TAB 2: PRACTICE QUIZ ---
    with tab2:
        st.header("❓ Practice Quiz")
        if st.button("Generate Questions 🎲", type="primary"):
            with st.spinner("Generating questions from PDF contents..."):
                questions = st.session_state.engine.generate_quiz(st.session_state.pdf_chunks)
                st.session_state.quiz_data = questions

        if "quiz_data" in st.session_state:
            questions = st.session_state.quiz_data
            if not questions:
                st.warning("Could not parse quiz JSON from the model. Try clicking 'Generate Questions' again.")
            else:
                for q in questions:
                    with st.container(border=True):
                        st.markdown(f"**Question {q.get('id', '')}:** {q.get('question', '')}")
                        options = q.get("options", [])
                        if options:
                            user_ans = st.radio(f"Select answer for Q{q.get('id')}:", options, key=f"q_rad_{q.get('id')}")
                            if st.button(f"Submit Q{q.get('id')}", key=f"q_btn_{q.get('id')}"):
                                selected_idx = options.index(user_ans)
                                if selected_idx == q.get("correct_index", 0):
                                    st.success("🎉 Correct!")
                                else:
                                    st.error("❌ Incorrect")
                                st.info(f"**Explanation:** {q.get('explanation', '')}")

    # --- TAB 3: DOCUMENT Q&A ---
    with tab3:
        st.header("💬 Ask Questions About the PDF")
        user_query = st.text_input("Ask any question regarding your uploaded document:")
        
        if st.button("Search & Answer 🔍", type="primary"):
            if user_query:
                with st.spinner("Searching document passages and generating answer..."):
                    context_chunks = st.session_state.engine.get_relevant_chunks(
                        user_query, st.session_state.pdf_chunks
                    )
                    answer = st.session_state.engine.answer_question(user_query, context_chunks)
                    st.session_state.chat_history.append({
                        "query": user_query, 
                        "answer": answer, 
                        "sources": context_chunks
                    })

        # Display Chat History
        if st.session_state.chat_history:
            st.markdown("---")
            for item in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(item["query"])
                with st.chat_message("assistant"):
                    st.markdown(item["answer"])
                    with st.expander("🔍 View Retrieved PDF Source Passages"):
                        for idx, src in enumerate(item["sources"]):
                            st.caption(f"**Passage {idx+1}:** {src}")

if __name__ == "__main__":
    main()