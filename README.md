# Intelligent Document Learning Assistant

A local Streamlit application that helps you study documents with a locally hosted Ollama language model. It supports PDF, Word (`.docx`), and PowerPoint (`.pptx`) files, and includes document summaries, practice quizzes, question-paper generation, document Q&A, and a quiz-performance dashboard.

## Features

- **PDF, DOCX, and PPTX reading**
  - PDF: extracts text page by page.
  - DOCX: extracts paragraphs and tables.
  - PPTX: extracts slide text, tables, and speaker notes when available.
- **Topic-by-topic summary** with an overview, sections, key points, and takeaways.
- **Practice quiz** generation with multiple-choice questions and explanations.
- **Question paper generator** for 5 one-mark, 3 three-mark, and 3 five-mark questions.
- **Document Q&A** using relevant text chunks retrieved with sentence embeddings.
- **Quiz performance dashboard** to track attempts and accuracy during the current app session.
- **Local model configuration** through the sidebar's Ollama model name setting.
- **Optional CUDA usage** for the Sentence Transformers embedding model when PyTorch detects a compatible NVIDIA GPU.

## Requirements

- Python 3.10 or later is a reasonable starting point.
- Ollama installed and running locally.
- An Ollama model installed locally, such as `llama3.2:1b`.
- A working NVIDIA driver and a CUDA-enabled PyTorch installation if you want GPU acceleration for embeddings.

## Installation

Open a terminal in the folder containing `app.py`.

Create and activate a virtual environment (recommended):

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install the application dependencies:

```bat
python -m pip install --upgrade pip
pip install streamlit pypdf python-docx python-pptx sentence-transformers scikit-learn numpy requests
```

Install PyTorch according to your machine and desired CUDA setup. Use the official selector to choose the suitable Windows/Pip/CUDA option:

https://pytorch.org/get-started/locally/

> A CUDA-enabled PyTorch package by itself does not guarantee GPU use. You also need a compatible NVIDIA GPU, a suitable NVIDIA driver, and a compatible PyTorch build.

## Set up Ollama

1. Install Ollama: https://ollama.com/
2. Download the model used by default:

   ```bat
   ollama pull llama3.2:1b
   ```

3. Make sure Ollama is running. The app sends requests to:

   ```text
   http://localhost:11434/api/generate
   ```

If you choose another model in the app sidebar, download that model first with `ollama pull <model-name>`.

## Run the application

From the project folder, run:

```bat
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## How to use it

1. Start Ollama and confirm that the selected model is available.
2. Run the Streamlit application.
3. Upload a PDF, DOCX, or PPTX file in the sidebar.
4. Click **Process Document**.
5. Choose one of the tabs:
   - **Summary** — generates a structured overview and topic breakdown.
   - **Practice Quiz** — generates multiple-choice questions.
   - **Question Paper** — creates a paper with the configured 1-, 3-, and 5-mark sections.
   - **Q&A** — asks questions about the uploaded document and can show retrieved source passages.
   - **Performance Dashboard** — reviews quiz attempts and accuracy.

## Check GPU detection for embeddings

Run this in the same Python environment used to launch Streamlit:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You can also check whether the NVIDIA driver can see the GPU:

```bat
nvidia-smi
```

The app selects the embedding device automatically:

```python
 device = "cuda" if torch.cuda.is_available() else "cpu"
 self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
```

If the diagnostic says `CUDA available: False`, the embedding model runs on CPU. Check that `nvidia-smi` works, install/update the NVIDIA driver if necessary, and reinstall PyTorch using the official installation selector.

## Understand the GPU distinction

The application has two distinct model workloads:

- **Sentence Transformers (`all-MiniLM-L6-v2`)** creates embeddings for retrieval. The app explicitly chooses CUDA if `torch.cuda.is_available()` returns `True`; otherwise, it uses the CPU.
- **Ollama** generates summaries, quiz questions, question papers, and answers. Ollama manages its own CPU/GPU allocation separately from PyTorch. A CUDA-enabled PyTorch installation does not automatically make Ollama use the GPU.

To inspect Ollama's current model processing and GPU/CPU allocation, run:

```bat
ollama ps
```

## Performance notes

- Long documents take longer to summarize because the app divides the text into blocks, summarizes each block, and then asks the model to consolidate those summaries.
- Smaller local models may be faster but can provide less detailed or less reliable summaries.
- The current Q&A retrieval function re-encodes the document chunks on each query. Caching chunk embeddings would reduce repeated work for large documents.
- GPU acceleration can help embedding workloads, but summary-generation speed depends heavily on Ollama's model, hardware allocation, context size, and the number of generation requests.
- Image-only/scanned PDFs may not yield readable text because the current PDF extraction uses text extraction and does not run OCR.

## Troubleshooting

### `CUDA available: False`

1. Run `nvidia-smi`.
2. Confirm that an NVIDIA GPU and driver are shown.
3. Check `torch.__version__` and `torch.version.cuda` in the same virtual environment used for Streamlit.
4. Install a compatible PyTorch build from https://pytorch.org/get-started/locally/.
5. Restart the terminal and Streamlit after installation.

### Ollama connection error

- Confirm Ollama is installed and running.
- Check that `http://localhost:11434` is reachable.
- Verify that the configured model exists using `ollama list`.
- Download a missing model with `ollama pull llama3.2:1b` or the model name selected in the sidebar.

### No text extracted

- Confirm that the file is a valid PDF, DOCX, or PPTX.
- Scanned PDFs may require OCR, which is not implemented in the current code.
- Password-protected or damaged files may fail to parse.

## Main dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web interface |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX paragraph and table extraction |
| `python-pptx` | PPTX slide, table, and notes extraction |
| `sentence-transformers` | Text embeddings for retrieval |
| `scikit-learn` | Cosine similarity |
| `numpy` | Vector operations |
| `requests` | Calls the local Ollama API |
| `torch` | Detects and runs the embedding model on CUDA when available |
