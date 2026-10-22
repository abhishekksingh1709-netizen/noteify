"""Noteify - a Gemma 4 study companion that answers only from your own notes.

Flow: paste/upload notes -> Noteify summary -> ask a question -> inspect evidence
      -> practice quiz -> feedback.

Run:  streamlit run app.py
Local:  Ollama running with Gemma 4 pulled (ollama pull gemma4:e4b)
Hosted: set BACKEND=gemini and GEMINI_API_KEY (Gemma 4 through the Gemini API)
"""
import json
import os
import re

import requests
import streamlit as st

# BACKEND: "ollama" (local laptop, default) or "gemini" (hosted Gemma 4 via Google AI Studio API)
BACKEND = os.getenv("BACKEND", "ollama").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv(
    "GEMMA_MODEL", "gemma-4-26b-a4b-it" if BACKEND == "gemini" else "gemma4:e4b"
)
NOT_ENOUGH = "The notes do not contain enough information to answer this."

SAMPLE_NOTES = """Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into chemical energy stored in glucose.
It takes place mainly in the chloroplasts of plant cells. Chloroplasts contain a green pigment called chlorophyll, which absorbs light, mostly red and blue wavelengths, and reflects green light.

The overall equation is: carbon dioxide + water + light energy -> glucose + oxygen. Plants take in carbon dioxide through tiny pores in their leaves called stomata, and absorb water through their roots.

Photosynthesis has two stages. In the light-dependent reactions, which happen in the thylakoid membranes, light energy splits water molecules, releasing oxygen and producing ATP and NADPH. In the Calvin cycle, which happens in the stroma, ATP and NADPH are used to convert carbon dioxide into glucose.

Several factors affect the rate of photosynthesis: light intensity, carbon dioxide concentration and temperature. If any one of these is in short supply, it becomes the limiting factor and slows the whole process. Very high temperatures can denature the enzymes involved and stop photosynthesis.
"""

STOPWORDS = set(
    "a an the of to in on at for and or is are was were be been what which who whom "
    "how why when where does do did it its this that these those with from by as "
    "can could should would about into than then there their they them you your i".split()
)


# ----------------------------------------------------------------------------
# Helpers: notes, chunks, retrieval, evidence checking
# ----------------------------------------------------------------------------
def norm(text: str) -> str:
    """Lowercase and strip punctuation/whitespace so quotes can be matched reliably."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def make_chunks(text: str):
    """Split notes into small numbered passages (P1, P2, ...)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > 350:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    return [{"id": f"P{i + 1}", "text": c} for i, c in enumerate(chunks)]


def retrieve(question: str, chunks, k: int = 6):
    """Simple keyword-overlap retrieval (no extra model needed)."""
    q = {w for w in norm(question).split() if w not in STOPWORDS}
    scored = []
    for i, c in enumerate(chunks):
        words = set(norm(c["text"]).split())
        scored.append((len(q & words), -i, c))
    scored.sort(reverse=True)
    top = [c for _, _, c in scored[:k]]
    return sorted(top, key=lambda c: int(c["id"][1:]))  # keep original order


def quote_in_notes(quote: str, notes: str) -> bool:
    q = norm(quote)
    return len(q) >= 8 and q in norm(notes)


def read_upload(file) -> str:
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(file)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


# ----------------------------------------------------------------------------
# Gemma 4 calls (through Ollama's local API)
# ----------------------------------------------------------------------------
def _parse_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences or extra text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("Model did not return JSON")
        return json.loads(m.group(0))


def _call_ollama(system: str, user: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_gemini(system: str, user: str) -> str:
    """Hosted Gemma 4 through the Gemini API (REST). System text is merged into the prompt."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=120,
    )
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def llm_json(system: str, user: str) -> dict:
    text = _call_gemini(system, user) if BACKEND == "gemini" else _call_ollama(system, user)
    return _parse_json(text)


def noteify_summary(notes: str) -> dict:
    system = (
        "You are a study assistant. Use ONLY the student's notes. Never add facts that "
        "are not in the notes. Write in plain, simple language a school student can follow. "
        'Reply with JSON only: {"title": str, "key_points": [str, ...], "simple_explanation": str}. '
        "Give 4 to 6 short key_points and a simple_explanation of at most 4 sentences."
    )
    return llm_json(system, f"NOTES:\n{notes[:12000]}")


def answer_question(question: str, chunks, notes: str) -> dict:
    ctx_chunks = retrieve(question, chunks)
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in ctx_chunks)
    system = (
        "You answer a student's question using ONLY the numbered passages given. "
        "Do NOT use outside knowledge, even if you know the answer. "
        "If the passages do not clearly contain the answer, set supported to false. "
        'Reply with JSON only: {"supported": bool, "answer": str, '
        '"evidence": [{"id": "P1", "quote": "exact words copied from that passage"}]}. '
        "Quotes must be copied word for word from the passages. Keep the answer short and simple."
    )
    data = llm_json(system, f"PASSAGES:\n{context}\n\nQUESTION: {question}")
    # Safeguard: keep only quotes that really appear in the notes.
    verified = []
    for ev in data.get("evidence", []) or []:
        quote = str(ev.get("quote", ""))
        if quote_in_notes(quote, notes):
            verified.append({"id": str(ev.get("id", "?")), "quote": quote})
    supported = bool(data.get("supported")) and len(verified) > 0
    return {
        "supported": supported,
        "answer": str(data.get("answer", "")) if supported else NOT_ENOUGH,
        "evidence": verified if supported else [],
        "passages": ctx_chunks,
    }


def generate_quiz(notes: str, difficulty: str, n: int = 4, focus=None) -> list:
    system = (
        "You write multiple-choice practice questions using ONLY the student's notes. "
        f"Difficulty: {difficulty}. Write exactly {n} questions with 4 options each. "
        'Reply with JSON only: {"questions": [{"question": str, "options": [str, str, str, str], '
        '"correct_index": 0-3, "hint": str, "explanation": str, '
        '"evidence": "exact words copied from the notes that prove the answer"}]}. '
        "The hint must nudge without giving the answer away. The explanation must say why the "
        "correct option is right and why a common wrong option is wrong."
    )
    extra = ""
    if focus:
        extra = "\nFocus on ideas related to these questions the student got wrong:\n- " + "\n- ".join(focus)
    data = llm_json(system, f"NOTES:\n{notes[:12000]}{extra}")
    quiz = []
    for q in data.get("questions", []):
        opts = q.get("options", [])
        idx = q.get("correct_index")
        if len(opts) == 4 and isinstance(idx, int) and 0 <= idx <= 3:
            q["evidence_ok"] = quote_in_notes(str(q.get("evidence", "")), notes)
            quiz.append(q)
    return quiz


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Noteify", page_icon="📝", layout="wide")
st.title("📝 Noteify")
st.caption("Your notes, simplified. Answers come only from your notes, with proof.")

ss = st.session_state
for key, default in {
    "notes": "", "chunks": [], "summary": None, "qa": [], "quiz": [], "checked": {}, "missed": [],
}.items():
    ss.setdefault(key, default)

with st.sidebar:
    st.header("1. Your notes")
    st.caption(f"Model: `{MODEL}` via {'Gemini API' if BACKEND == 'gemini' else 'Ollama'}")
    pasted = st.text_area("Paste notes (one topic, a few pages)", height=220, key="paste_box")
    upload = st.file_uploader("...or upload .txt / .pdf", type=["txt", "pdf"])
    col_a, col_b = st.columns(2)
    load_clicked = col_a.button("Load notes", type="primary")
    sample_clicked = col_b.button("Use sample")

    new_text = None
    if sample_clicked:
        new_text = SAMPLE_NOTES
    elif load_clicked:
        new_text = read_upload(upload) if upload else pasted
    if new_text is not None:
        if len(new_text.strip()) < 40:
            st.error("Please add a few sentences of notes first.")
        else:
            ss.notes = new_text.strip()
            ss.chunks = make_chunks(ss.notes)
            ss.summary, ss.qa, ss.quiz, ss.checked, ss.missed = None, [], [], {}, []
            st.success(f"Loaded {len(ss.chunks)} passages.")

    if ss.notes:
        with st.expander("Notes loaded (passages)"):
            for c in ss.chunks:
                st.markdown(f"**{c['id']}** {c['text']}")

if not ss.notes:
    st.info("Paste or upload notes in the sidebar, or click **Use sample**, to begin.")
    st.stop()

tab_sum, tab_ask, tab_quiz = st.tabs(["✨ Noteify summary", "❓ Ask your notes", "🧪 Practice quiz"])

# --- Summary -----------------------------------------------------------------
with tab_sum:
    if st.button("Simplify my notes"):
        with st.spinner("Gemma 4 is simplifying your notes..."):
            try:
                ss.summary = noteify_summary(ss.notes)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not reach the model: {e}")
    if ss.summary:
        st.subheader(ss.summary.get("title", "Summary"))
        for p in ss.summary.get("key_points", []):
            st.markdown(f"- {p}")
        st.info(ss.summary.get("simple_explanation", ""))

# --- Ask ---------------------------------------------------------------------
with tab_ask:
    question = st.text_input("Ask a question about your notes")
    if st.button("Ask") and question.strip():
        with st.spinner("Searching your notes..."):
            try:
                res = answer_question(question.strip(), ss.chunks, ss.notes)
                ss.qa.insert(0, {"q": question.strip(), **res})
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not reach the model: {e}")
    for item in ss.qa:
        st.markdown(f"**Q:** {item['q']}")
        if item["supported"]:
            st.success(item["answer"])
            st.markdown("**Evidence from your notes:**")
            for ev in item["evidence"]:
                st.markdown(f"> “{ev['quote']}”  \n> *(passage {ev['id']})*")
        else:
            st.warning(
                "⚠️ " + item["answer"]
                + " Try rephrasing, or add the missing material to your notes."
            )
        st.divider()

# --- Quiz --------------------------------------------------------------------
with tab_quiz:
    c1, c2, c3 = st.columns([1, 1, 2])
    difficulty = c1.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    if c2.button("Generate quiz"):
        with st.spinner("Writing practice questions..."):
            try:
                ss.quiz = generate_quiz(ss.notes, difficulty)
                ss.checked, ss.missed = {}, []
                if not ss.quiz:
                    st.error("The model returned no usable questions. Try again.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not reach the model: {e}")
    if ss.missed and c3.button("Quiz me on what I missed (easier)"):
        with st.spinner("Writing focused questions..."):
            try:
                ss.quiz = generate_quiz(ss.notes, "easy", focus=list(ss.missed))
                ss.checked, ss.missed = {}, []
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not reach the model: {e}")

    score = 0
    for i, q in enumerate(ss.quiz):
        st.markdown(f"**Q{i + 1}. {q['question']}**")
        choice = st.radio(
            "Choose an answer", q["options"], index=None, key=f"choice_{i}", label_visibility="collapsed"
        )
        b1, b2 = st.columns([1, 1])
        if b1.button("Check answer", key=f"check_{i}") and choice is not None:
            ss.checked[i] = q["options"].index(choice) == q["correct_index"]
            if not ss.checked[i] and q["question"] not in ss.missed:
                ss.missed.append(q["question"])
        if b2.button("Hint", key=f"hint_{i}"):
            st.info("💡 " + str(q.get("hint", "Re-read the related part of your notes.")))
        if i in ss.checked:
            if ss.checked[i]:
                score += 1
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Not quite. Correct answer: **{q['options'][q['correct_index']]}**")
            st.markdown(f"**Why:** {q.get('explanation', '')}")
            if q.get("evidence_ok"):
                st.markdown(f"> From your notes: “{q['evidence']}”")
        st.divider()
    if ss.quiz and ss.checked:
        st.metric("Score", f"{score} / {len(ss.checked)} answered")
