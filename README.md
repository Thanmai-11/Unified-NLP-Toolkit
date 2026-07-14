# A Unified NLP Research Toolkit

**Classical NLP vs. pretrained Transformer methods, side-by-side, across the core language-understanding pipeline.**

A Streamlit app that runs classical/statistical NLP methods and pretrained Transformer models on the same input text and displays their outputs side-by-side, spanning preprocessing, spelling correction, language modeling, POS tagging, dependency parsing, distributional/lexical semantics, topic modeling, NER, summarization, and sentiment analysis.

**[Live demo →](https://YOUR-APP-NAME.streamlit.app)**

## What this compares

| Module | Classical / Statistical (trained live on your input) | Transformer (pretrained, inference only) |
|---|---|---|
| Preprocessing | NLTK tokenizer, Porter stemmer | POS-aware WordNet lemmatizer |
| Spelling correction | Norvig edit-distance corrector | — |
| Language modeling | Bigram model, Laplace smoothing | GPT-2 perplexity |
| POS tagging | NLTK Averaged Perceptron | spaCy neural pipeline |
| Parsing | — | spaCy dependency parser |
| Distributional semantics | GloVe static embeddings | BERT contextual embeddings |
| Lexical semantics | WordNet synsets | — |
| Topic modeling | LDA (gensim), trained live | — |
| NER / information extraction | — | spaCy NER |
| Summarization | TextRank (extractive) | DistilBART-CNN (abstractive) |
| Sentiment analysis | VADER (lexicon-based) | DistilBERT-SST2 |

Every classical component is trained from scratch, live, on whatever text you enter — not a fixed pretrained resource. Every Transformer component is used for inference only, with no fine-tuning, so results reflect general-purpose pretrained performance.

## Tech stack

Python · Streamlit · NLTK · spaCy · gensim · scikit-learn · Hugging Face Transformers · PyTorch

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/unified-nlp-toolkit.git
cd unified-nlp-toolkit
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4 \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng vader_lexicon words
export HF_TOKEN=hf_your_token_here   # free at huggingface.co/settings/tokens
streamlit run app.py
```

The spaCy English model is pinned in `requirements.txt` as a direct wheel, so no separate `spacy download` step is needed. `HF_TOKEN` isn't strictly required to run the app, but without it, model downloads from Hugging Face Hub are rate-limited per shared IP and can stall.

## Deploying your own copy

Deployed here on [Streamlit Community Cloud](https://share.streamlit.io) (free tier). To deploy your own:

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
