# A Unified NLP Research Toolkit
### Classical NLP to Modern Transformer-based Language Understanding

A single-page Streamlit app comparing classical/statistical NLP methods
against pretrained Transformer methods, across the full 12-week NPTEL NLP
syllabus. Built as a 1-credit mini project deliverable + portfolio demo.

**Scope note (read this first):** every "classical" component here is
trained live, from scratch, on whatever text you input — that's genuinely
your own model, your own result. Every Transformer component uses a
**pretrained** model for **inference only** — no fine-tuning. This is a
deliberate choice: it's what makes full syllabus coverage possible in days
rather than months on free-tier compute, and it still supports a real,
defensible comparison (how pretrained Transformer representations differ
from statistical baselines you built yourself). It does **not**, by itself,
constitute a novel research contribution — see "If you want to go further"
below for what that would actually require.

## Live demo

Deployed on Streamlit Community Cloud (free tier). Transformer-backed tabs
(Language Modeling, POS/Parsing, Distributional Semantics, NER,
Summarization, Sentiment) download their pretrained model checkpoints the
first time each is used in a session; this is cached via
`st.cache_resource`, so subsequent runs on the same tab are fast. Expect
the first click on each transformer tab to take anywhere from a few
seconds to a minute or two.

**Note on memory:** Streamlit Community Cloud's free tier caps apps at
1 GB RAM. This app is designed so each transformer model only loads when
its tab's button is actually clicked, but if a single session runs through
every heavy tab back-to-back, cached models can accumulate and approach
that limit. If you see a memory error, restart the app from the Streamlit
Cloud dashboard (Manage app → Reboot).

## Setup

### Option A: Run locally
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4 \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng vader_lexicon words
streamlit run app.py
```
(The spaCy English model is pinned directly in `requirements.txt` as a
wheel URL, so a separate `python -m spacy download en_core_web_sm` step is
not required — though it's harmless to run if you prefer it.)

### Option B: Run on Google Colab (free GPU)
```python
# Cell 1
!pip install -q streamlit nltk spacy sumy transformers torch scikit-learn pyngrok
!python -m spacy download en_core_web_sm -q

# Cell 2 - upload app.py to Colab's file browser first, then:
!streamlit run app.py &>/content/logs.txt &

# Cell 3 - expose the local Streamlit port publicly
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(public_url)
```
(You'll need a free ngrok auth token — sign up at ngrok.com, then run
`!ngrok config add-authtoken YOUR_TOKEN` before Cell 3.)

### Option C: Deploy publicly on Streamlit Community Cloud (recommended for sharing)
1. Push this repo (`app.py`, `requirements.txt`, this `README.md`) to a
   **public** GitHub repository.
2. Go to **share.streamlit.io**, sign in with GitHub.
3. Click **New app** → pick the repo, branch (`main`), and set
   **Main file path** to `app.py`.
4. Click **Deploy**. The build takes a few minutes on first run (it's
   installing torch, transformers, spacy, etc.) — watch the logs
   in the browser.
5. Once it's live, you get a public URL
   (`https://YOUR-APP-NAME.streamlit.app`) that anyone can open, no login
   required.
6. If you hit the 1 GB memory ceiling during heavy use, use **Manage app**
   (bottom right of the running app) → **Reboot app** to clear cached
   models and free memory.

## Day-by-day plan (for a <1 week timeline)

**Day 1:** Get `app.py` running locally or on Colab. Tabs 1, 2, 4, 8, 12
(preprocessing, spelling, POS, WordNet, sentiment) need no heavy downloads
and should work immediately — confirm the environment before anything else.

**Day 2:** Bring up the transformer-backed tabs one at a time (GPT-2
perplexity, spaCy parsing/NER, BERT embeddings, DistilBERT sentiment,
DistilBART summarization). Each `st.cache_resource` model loads once —
budget ~5-10 min of Colab GPU time total for all downloads combined.

**Day 3:** Run the app against 3-4 different input texts (a news article,
a review, a Wikipedia paragraph, a piece of technical writing) and record
outputs from every tab as your experimental evidence. Screenshot or export
each comparison — this is your raw results section.

**Day 4:** Write the report (see `REPORT_OUTLINE.md`). Draft the
comparison tables and the discussion of where classical methods hold up
vs. where Transformers clearly win (WordNet vs. BERT context-sensitivity is
your strongest, most concrete example — it's directly measurable via the
cosine similarity tab).

**Day 5:** Polish: clean up the GitHub repo, write the README (this one),
proofread the report, do a final run-through of the live app to make sure
nothing breaks in front of an evaluator. Deploy to Streamlit Community
Cloud (Option C above) once you're happy with it, so you have a public
link to share alongside the report.

**Day 6-7 (buffer):** Fix whatever broke on Day 5. Reserve this — don't
plan to need it, but you will.

## If you want to go further (after submission)

A real novel contribution — the kind that could genuinely target a
workshop or, later, a conference — would need one focused idea, not full
breadth. Candidates worth discussing once this is submitted:
- A hybrid symbolic-Transformer pipeline for one specific task
- An efficiency-vs-accuracy analysis with real profiling (latency, memory,
  parameter count) across model tiers on a held-out benchmark
- A targeted probe of whether attention patterns correlate with dependency
  parse structure (testable with the parsing + embeddings tabs you already
  have as a base)

Each of these is a multi-month project on its own. Worth a separate
conversation once the course deliverable is done.
1. Fork/push this repo to GitHub.
2. On share.streamlit.io, create a new app pointing at `app.py`.
3. Under **Advanced settings → Secrets**, add `HF_TOKEN = "hf_your_token_here"`.
4. Deploy.

See `.gitignore` — secrets are never committed to this repo; see the token's own settings page to generate one.

## Memory design note

Streamlit Community Cloud's free tier caps apps at 1 GB RAM. This app's four largest models (GPT-2, BERT, DistilBERT-SST2, DistilBART-CNN) would need roughly 2.5–2.8 GB combined if all cached at once, so only one of the four stays resident in memory at a time — loading a different one evicts whichever was previously loaded. A sidebar indicator shows which model is currently loaded. This affects memory usage only, not model outputs.

## Scope

This project demonstrates a full-pipeline comparison across a standard NLP syllabus; it is not, on its own, a novel research contribution. See the accompanying report for a discussion of where classical methods hold up against Transformer baselines and where they don't, along with proposed directions (hybrid symbolic-Transformer pipelines, efficiency-vs-accuracy profiling, attention/dependency correlation studies) for extending this into original research.

## License

MIT
