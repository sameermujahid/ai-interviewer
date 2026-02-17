# AI Interviewer — RAG + SLM Powered Interview Preparation

AI Interviewer is a Flask-based web application that generates personalized interview questions and simulates interview answers using Retrieval-Augmented Generation (RAG) and a Small Language Model (SLM).

The system analyzes a candidate’s resume and target job role to produce:

* 10 Technical Interview Questions
* 10 Behavioral Interview Questions
* AI-simulated interview answers grounded strictly in the uploaded resume

The application uses embeddings, vector search, and a causal language model to create tailored and context-aware outputs.

---

## Features

### Resume Upload and Extraction

* Supports PDF and DOCX formats
* Extracts and cleans resume text
* Displays extracted content on the frontend

### RAG-Based Question Generation

* Resume is chunked and embedded
* Stored in ChromaDB vector database
* Relevant context retrieved per query
* Generates:

  * 10 technical questions
  * 10 behavioral questions
* Strict formatting control

### Interview Simulation

* Accepts interviewer question input
* Uses full resume context
* Generates realistic, professional responses
* Avoids hallucination by grounding strictly in resume

---

## Architecture Overview

Frontend:

* HTML
* CSS
* JavaScript

Backend:

* Flask

RAG Stack:

* SentenceTransformers (Jina embeddings v3 with fallback)
* ChromaDB (persistent vector store)
* Hugging Face Transformers
* Qwen2.5-0.5B-Instruct (CPU inference)

Document Processing:

* PyMuPDF (PDF extraction)
* python-docx (DOCX extraction)

---

## Project Structure

```
ai-interviewer/
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── main.js
├── requirements.txt
└── README.md
```

---

## Installation (Local Development)

### 1. Clone Repository

```
git clone https://github.com/sameermujahid/ai-interviewer
cd ai-interviewer
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run Application

```
python app.py
```

Open in browser:

```
http://localhost:5000
```

---

The application is available at:

```
https://huggingface.co/spaces/sameer-mujahid/ai-interviewer
```

---

## How It Works

### 1. Resume Processing

* Resume is extracted and cleaned
* Text is chunked into manageable segments
* Each chunk is embedded using a transformer-based embedding model

### 2. Vector Storage

* Embeddings stored in ChromaDB
* Cosine similarity used for retrieval

### 3. Retrieval-Augmented Generation

* Relevant resume chunks retrieved per task
* Context injected into structured LLM prompt
* Model generates strictly formatted output

### 4. Interview Simulation

* Full resume context passed to model
* Model instructed not to hallucinate
* Produces natural candidate-style answers

---

## Configuration

Environment variables:

```
HF_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
CHROMA_DIR=/tmp/ai_interviewer_chroma
PORT=5000
```

---

## Design Principles

* Deterministic output (low temperature)
* Strict formatting enforcement
* Resume-grounded generation
* CPU-compatible inference
* Modular RAG pipeline
* Clear separation between retrieval and generation

---


## Use Cases

* Interview preparation
* Resume-based mock interviews
* AI portfolio project
* Demonstration of RAG pipeline implementation
* Educational tool for LLM + vector DB integration


