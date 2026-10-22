# Noteify - Gemma 4 study companion

Paste or upload notes on one topic. Noteify simplifies them, answers questions
using only those notes (with quoted evidence), says so when the notes don't
contain the answer, and quizzes you with hints and explanations.

## Run (about 5 minutes)

1. Install Ollama: https://ollama.com/download
2. Pull the model (about 9.6 GB):  `ollama pull gemma4:e4b`
3. Install dependencies:           `pip install -r requirements.txt`
4. Start the app:                  `streamlit run app.py`

If your laptop is slow, try `set GEMMA_MODEL=gemma4:e2b` (Windows) or
`export GEMMA_MODEL=gemma4:e2b` (Mac/Linux) before step 4.

## Demo checklist (matches the problem statement)

1. Click **Use sample** (photosynthesis notes), then **Simplify my notes**.
2. Ask: "Where does photosynthesis take place?" -> answer + quoted evidence.
3. Quiz tab -> **Generate quiz** -> pick a wrong option -> **Check answer** to see the explanation. Use **Hint** before answering.
4. Ask: "Who discovered photosynthesis?" -> the notes don't say, so Noteify shows the "not enough information" warning.

## How output quality is controlled

- Prompts forbid outside knowledge; answers must quote the notes.
- Every quote is checked against the note text in code. No valid quote -> "not enough information".
- Quiz answers are graded by code, not by the model; the model only writes the question, hint and explanation.
