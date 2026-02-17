#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Interviewer: Flask app (CPU version with debug logging)
- Upload resume (PDF/DOCX)
- Extract text and show on frontend
- Generate 10 technical + 10 behavioral interview questions
- Uses RAG + SLM
"""

import os
import re
import uuid
import tempfile
from typing import List, Tuple

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ----------- Document extraction -----------
import fitz  # PyMuPDF
import docx

# ----------- Embeddings / Vector DB -----------
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ----------- Text generation (SLM) -----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------- Config --------------------
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL_PRI = "jinaai/jina-embeddings-v3"
EMBED_MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(tempfile.gettempdir(), "ai_interviewer_chroma"))

ALLOWED_EXT = {".pdf", ".docx"}

app = Flask(__name__)

# ----------------- Helpers -------------------
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

# ----------------- Models --------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            print(f"Loading primary embedder: {EMBED_MODEL_PRI}")
            _embedder = SentenceTransformer(EMBED_MODEL_PRI, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Failed to load {EMBED_MODEL_PRI}: {e}")
            print(f"➡️ Falling back to {EMBED_MODEL_FALLBACK}")
            _embedder = SentenceTransformer(EMBED_MODEL_FALLBACK)
    return _embedder

_generation_pipe = None
def get_generator():
    global _generation_pipe
    if _generation_pipe is None:
        print(f"Loading generator model: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float32,   # ✅ CPU only
            device_map=None
        ).to("cpu")
        _generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            do_sample=False,            # deterministic output (IMPORTANT)
            temperature=0.01,           # encourages stable formatting
            repetition_penalty=1.1,
            device=-1
        )

    return _generation_pipe

# --------------- Vector Store ----------------
def build_resume_collection(resume_text: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    collection_name = f"resume_{uuid.uuid4().hex[:8]}"
    col = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    chunks = chunk_text(resume_text, max_chars=900)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    ids = [f"c_{i}" for i in range(len(chunks))]
    col.add(documents=chunks, embeddings=embeddings.tolist(), ids=ids)
    return col

def retrieve(col, query: str, k: int = 5) -> List[str]:
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    res = col.query(query_embeddings=q_emb.tolist(), n_results=k)
    return res.get("documents", [[]])[0]

# --------------- Prompting -------------------
SYS_PROMPT = """
You are an expert technical interviewer. 
Your job is to generate EXACTLY 10 technical and 10 behavioral interview questions.

RULES:
- ALWAYS output exactly 10 questions for each section.
- Do NOT generate MCQs.
- Only open-ended questions.
- One question per line.
- No numbering inside the question text (the number is only a prefix).
- NO explanations, NO extra text, NO commentary.
- Questions must be tailored to the candidate's resume context and target role.
"""
TECHNICAL_INSTRUCTION = """
Generate EXACTLY 10 technical interview questions.
Number them 1 through 10.
Each question must be a single line.
Tailor the questions strictly to the candidate's resume context.
"""

BEHAVIORAL_INSTRUCTION = """
Generate EXACTLY 10 behavioral interview questions.
Number them 1 through 10.
Each question must be a single line.
Questions must involve teamwork, conflict resolution, leadership, delivery,
problem solving, ownership, deadlines, pressure handling, and cross-team collaboration.
Tailor them strictly to the candidate's resume context.
"""

def build_prompt(job_title: str, context_snippets: List[str], mode: str) -> str:
    ctx = "\n\n".join([f"- {c}" for c in context_snippets])
    instruction = TECHNICAL_INSTRUCTION if mode == "technical" else BEHAVIORAL_INSTRUCTION
    user_block = f"""Target Role: {job_title}

Candidate Resume Context:
{ctx}

Task: {instruction}
"""
    prompt = f"<|system|>\n{SYS_PROMPT}\n</|system|>\n<|user|>\n{user_block}\n</|user|>\n<|assistant|>\n"""

    # ✅ Print what is being sent to the LLM
    print("\n================ LLM PROMPT DEBUG ================")
    print(f"Job Title: {job_title}")
    print(f"Mode: {mode}")
    print(f"Prompt Sent to LLM:\n{prompt}")
    print("=================================================\n")

    return prompt
def parse_lines(s: str) -> List[str]:
    # Extract assistant-only section
    if "<|assistant|>" in s:
        s = s.split("<|assistant|>")[-1]

    # Remove extra whitespace
    s = s.strip()

    # Split into lines
    lines = s.split("\n")

    clean = []
    for ln in lines:
        ln = ln.strip()

        # Remove "1.", "1)", "1-" etc.
        ln = re.sub(r"^\s*\d+[\.\-\)\:]\s*", "", ln)

        # Filter junk
        if len(ln) < 4:
            continue
        if ln.lower().startswith(("assistant", "system", "user")):
            continue

        clean.append(ln)

    # Safety: If less than 10 questions, we add placeholders
    if len(clean) < 10:
        while len(clean) < 10:
            clean.append("⚠️ Missing question — model did not generate enough output.")

    return clean[:10]
def generate_questions(job_title: str, resume_text: str) -> Tuple[List[str], List[str]]:
    # Retrieve contextual snippets
    col = build_resume_collection(resume_text)

    tech_ctx = retrieve(col, f"technical interview topics for {job_title}", k=5)
    beh_ctx  = retrieve(col, f"behavioral and soft skills situations for {job_title}", k=5)

    pipe = get_generator()

    # Build prompts
    tech_prompt = build_prompt(job_title, tech_ctx, mode="technical")
    beh_prompt  = build_prompt(job_title, beh_ctx, mode="behavioral")

    # Generate raw text
    tech_out = pipe(tech_prompt)[0]["generated_text"]
    beh_out  = pipe(beh_prompt)[0]["generated_text"]

    # Parse
    tech_q = parse_lines(tech_out)
    beh_q  = parse_lines(beh_out)

    # Safety check: re-generate if missing questions
    if len(tech_q) < 10:
        tech_out = pipe(tech_prompt)[0]["generated_text"]
        tech_q = parse_lines(tech_out)

    if len(beh_q) < 10:
        beh_out = pipe(beh_prompt)[0]["generated_text"]
        beh_q = parse_lines(beh_out)

    return tech_q[:10], beh_q[:10]


# ----------------- Routes --------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF or DOCX files are allowed"}), 400

    filename = secure_filename(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(tmp_path)
        else:
            text = extract_text_from_docx(tmp_path)
        text = clean_text(text)
        if len(text) > 200_000:
            text = text[:200_000]
        print("\n✅ Extracted Resume Text (first 500 chars):\n", text[:500], "...\n")
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {e}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    job_title = (data.get("jobTitle") or "").strip()
    resume_text = (data.get("resumeText") or "").strip()

    if not job_title:
        return jsonify({"error": "jobTitle is required"}), 400
    if not resume_text:
        return jsonify({"error": "resumeText is empty"}), 400

    try:
        tech_q, beh_q = generate_questions(job_title, resume_text)
        return jsonify({"technical": tech_q, "behavioral": beh_q})
    except Exception as e:
        return jsonify({"error": f"Failed to generate questions: {e}"}), 500
@app.route("/simulate_answer", methods=["POST"])
def simulate_answer():
    """
    Interview Simulation (Full Resume Context):
    - Input: jobTitle, resumeText, interviewerQuestion
    - Output: Candidate-like answer grounded in **entire resume**
    """
    data = request.get_json(force=True)
    job_title = (data.get("jobTitle") or "").strip()
    resume_text = (data.get("resumeText") or "").strip()
    interviewer_question = (data.get("interviewerQuestion") or "").strip()

    if not job_title:
        return jsonify({"error": "jobTitle is required"}), 400
    if not resume_text:
        return jsonify({"error": "resumeText is empty"}), 400
    if not interviewer_question:
        return jsonify({"error": "interviewerQuestion is required"}), 400

    try:
        print("\n================ INTERVIEW SIMULATION DEBUG ================")
        print(f"Job Title: {job_title}")
        print(f"Interviewer Question: {interviewer_question}")
        print(f"Resume Text Length: {len(resume_text)} characters")
        print("===========================================================\n")

        # Build prompt using **full resume**
        prompt = f"""
<|system|>
You are simulating a job candidate in an interview. 
Your task: respond naturally, confidently, and professionally to the interviewer's question.
Use the entire resume text below as reference. Base your answers strictly on the resume.
Do NOT make up experiences, skills, or projects.
If the question asks about projects, list all relevant projects from the resume with 1-2 sentence descriptions each.
</|system|>

<|user|>
Target Role: {job_title}

Candidate Full Resume:
{resume_text}

Interviewer Question: {interviewer_question}
</|user|>
<|assistant|>
"""

        print("Prompt Sent to LLM:\n", prompt[:1000], "...")  # print first 1000 chars for debug

        # Generate response
        pipe = get_generator()
        output = pipe(prompt)[0]["generated_text"]

        # Extract assistant answer
        parts = output.split("<|assistant|>")
        answer = parts[-1].strip() if parts else output.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": f"Failed to simulate answer: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
