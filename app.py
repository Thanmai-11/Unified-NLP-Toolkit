"""
A Unified NLP Research Toolkit: Classical NLP to Modern Transformer-based
Language Understanding

Single-file Streamlit app. Each tab corresponds to a week of the NPTEL
NLP syllabus and shows a classical/statistical baseline side-by-side
with a pretrained transformer output, for direct comparison.

Design principle: classical/statistical components are trained LIVE on
whatever text the user provides (cheap, CPU-only, seconds). Transformer
components are loaded PRETRAINED and used for inference only (no
fine-tuning) — this keeps the whole app runnable on free-tier Colab/
Kaggle GPU or even CPU alone, while still covering the full syllabus.

Run locally:      streamlit run app.py
Run on Colab:      see README.md for the ngrok/localtunnel snippet
"""

import os

# Hard network timeout (seconds) for Hugging Face Hub downloads/reads.
# Without this, huggingface_hub's underlying `requests` calls have no
# read-timeout, so if the Hub is merely DEGRADED (slow CDN, partial outage)
# rather than fully unreachable, a from_pretrained() call can hang
# indefinitely instead of raising an error that _load_heavy()'s except-block
# below could catch and surface. Must be set before transformers/
# huggingface_hub is imported anywhere, since huggingface_hub reads this env
# var once at import time.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import streamlit as st
import re
import math
import time
from collections import Counter, defaultdict

st.set_page_config(page_title="Unified NLP Toolkit", layout="wide")

# ---------------------------------------------------------------------------
# Lazy-loaded heavy resources
# ---------------------------------------------------------------------------
# spaCy / NLTK / GloVe are small (tens of MB) and shared across many tabs, so
# they stay behind st.cache_resource and are kept for the whole session.
#
# GPT-2, BERT, DistilBERT-SST2, and DistilBART-CNN are each large enough
# (0.25-1.2 GB in memory) that caching all four simultaneously — which
# st.cache_resource would do by default, since each gets its own permanent
# cache slot — needs roughly 2.5-2.8 GB. That comfortably exceeds Streamlit
# Community Cloud's free-tier 1 GB memory cap, which causes the app to hang
# or silently restart rather than raise a clean error. _load_heavy() below
# keeps at most ONE of these four resident at a time: loading a different
# one evicts whichever was previously loaded. Model checkpoints and outputs
# are unchanged — this only changes how many stay in memory simultaneously.
# ---------------------------------------------------------------------------

import gc


# Substrings that indicate a transient network/Hub problem (worth retrying)
# rather than a real code or environment problem (not worth retrying).
_HF_TRANSIENT_ERRORS = (
    "timed out", "timeout", "connection", "504", "502", "503",
    "reset by peer", "temporarily unavailable", "max retries exceeded",
)


def _load_heavy(key, loader_fn, retries=2, backoff_seconds=5):
    slot = st.session_state.get("_heavy_model_slot")
    if slot is not None and slot[0] == key:
        return slot[1]  # this model is already resident, reuse it

    if slot is not None:
        st.session_state.pop("_heavy_model_slot", None)
        del slot
        gc.collect()

    last_err = None
    with st.spinner(f"Loading {key} (evicting any other heavy model first)..."):
        for attempt in range(retries + 1):
            try:
                resource = loader_fn()
                last_err = None
                break
            except Exception as e:
                last_err = e
                transient = any(s in str(e).lower() for s in _HF_TRANSIENT_ERRORS)
                if attempt < retries and transient:
                    time.sleep(backoff_seconds)
                    continue
                break

    if last_err is not None:
        st.error(
            f"Failed to load **{key}**: `{type(last_err).__name__}: {last_err}`\n\n"
            "This is usually one of: a missing dependency (e.g. `accelerate`), "
            "a Hugging Face Hub outage or degraded CDN (check "
            "status.huggingface.co), or a memory limit hit during model "
            "deserialization. Downloads now time out after 30s "
            "(`HF_HUB_DOWNLOAD_TIMEOUT`) and retry twice with backoff, so a "
            "stalled connection fails here with this message instead of "
            "hanging silently. Check the 'Manage app' logs for the full "
            "traceback, then retry."
        )
        st.stop()

    st.session_state["_heavy_model_slot"] = (key, resource)
    return resource


@st.cache_resource(show_spinner="Loading NLTK data...")
def load_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
                "vader_lexicon"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return nltk


@st.cache_resource(show_spinner="Loading spaCy pipeline...")
def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


def load_sentiment_pipeline():
    def _load():
        from transformers import pipeline
        return pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english",
                         model_kwargs={"low_cpu_mem_usage": True})
    return _load_heavy("distilbert-sst2", _load)


def load_summarizer():
    def _load():
        # Loading the model + tokenizer directly (rather than via
        # pipeline("summarization", ...)) so this doesn't break if a future
        # transformers version renames/removes the "summarization" task
        # string from its pipeline registry (this happened in transformers
        # 5.x, which dropped several task aliases including this one).
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "sshleifer/distilbart-cnn-12-6"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model.eval()
        return tok, model
    return _load_heavy("distilbart-cnn", _load)


def load_gpt2():
    def _load():
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2", low_cpu_mem_usage=True)
        model.eval()
        return tok, model
    return _load_heavy("gpt2", _load)


def load_bert():
    def _load():
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased", low_cpu_mem_usage=True)
        model.eval()
        return tok, model
    return _load_heavy("bert-base", _load)


DEFAULT_TEXT = (
    "Natural language processing enables computers to understand, interpret, "
    "and generate human language. Early systems relied on hand-crafted rules "
    "and symbolic grammars, while modern systems learn statistical patterns "
    "directly from large corpora. Transformer architectures, introduced in "
    "2017, replaced recurrent networks as the dominant approach for most "
    "language understanding tasks. Researchers continue to study whether "
    "these models truly understand syntax and semantics or merely exploit "
    "surface-level statistical regularities in their training data."
)

st.title("A Unified NLP Research Toolkit")
st.caption(
    "Classical / statistical NLP vs. pretrained Transformer methods, "
    "side-by-side, across the core language-understanding pipeline."
)

text_input = st.text_area("Input text (used across all tabs)", DEFAULT_TEXT, height=140)

_resident = st.session_state.get("_heavy_model_slot")
st.sidebar.caption(
    f"Heavy model in memory: **{_resident[0]}**" if _resident
    else "Heavy model in memory: none yet"
)
st.sidebar.caption(
    "Only one large transformer (GPT-2 / BERT / DistilBERT-SST2 / "
    "DistilBART) stays loaded at a time to fit the 1 GB free-tier memory "
    "limit — switching tabs that use a different one will trigger a "
    "reload."
)

tabs = st.tabs([
    "Preprocessing", "Spelling", "Language Modeling", "POS Tagging",
    "Parsing", "Distributional Semantics", "Lexical Semantics",
    "Topic Modeling", "NER / IE", "Summarization & Classification",
    "Sentiment Analysis"
])

# ---------------------------------------------------------------------------
# Tab 1: Preprocessing (Week 1)
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Text Preprocessing")
    nltk = load_nltk()
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords, wordnet as wn_pos_lookup
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk import pos_tag as pos_tag_for_lemma

    tokens = word_tokenize(text_input)
    sw = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def is_word(t):
        # Keep normal words AND hyphenated compounds (e.g. "hand-crafted"),
        # drop pure punctuation. isalpha() alone would drop the hyphenated
        # case entirely, silently losing legitimate vocabulary.
        return bool(re.fullmatch(r"[A-Za-z]+(-[A-Za-z]+)*", t))

    def penn_to_wordnet(tag):
        # WordNetLemmatizer defaults to noun-only lemmatization unless told
        # otherwise; mapping the POS tagger's output onto WordNet's simpler
        # tag set lets verbs/adjectives/adverbs lemmatize correctly too.
        if tag.startswith("V"):
            return wn_pos_lookup.VERB
        if tag.startswith("J"):
            return wn_pos_lookup.ADJ
        if tag.startswith("R"):
            return wn_pos_lookup.ADV
        return wn_pos_lookup.NOUN  # default / NN*, safest fallback

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tokenization")
        st.write(tokens[:40])
        st.subheader("Stopword removal")
        filtered = [t for t in tokens if t.lower() not in sw and is_word(t)]
        st.write(filtered[:40])
    with col2:
        st.subheader("Stemming (Porter)")
        st.write([stemmer.stem(t) for t in filtered[:40]])
        st.subheader("Lemmatization (WordNet, POS-aware)")
        tagged = pos_tag_for_lemma(filtered[:40])
        st.write([lemmatizer.lemmatize(t, pos=penn_to_wordnet(tag)) for t, tag in tagged])
        st.caption(
            "Now uses each word's POS tag (from the POS Tagging tab's same "
            "tagger) so verbs, adjectives, and adverbs lemmatize correctly — "
            "not just nouns, which is WordNetLemmatizer's silent default."
        )

# ---------------------------------------------------------------------------
# Tab 2: Spelling correction (Week 2)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Spelling Correction")
    st.write("Norvig-style edit-distance corrector (classical baseline).")
    misspelled = st.text_input("Type a word to correct", "langauge")

    @st.cache_resource(show_spinner="Building spelling correction vocabulary...")
    def build_norvig_model(_text_for_cache_key):
        # Frequency model: the shared input text (weighted higher, since it's
        # the relevant domain) plus NLTK's ~235k word English list for
        # general coverage. _text_for_cache_key forces Streamlit to rebuild
        # this whenever the input text actually changes, instead of caching
        # forever after the first run.
        import nltk
        words = re.findall(r"[a-z]+", _text_for_cache_key.lower()) * 10  # upweight
        dictionary_loaded = False
        try:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
            wl = nltk_words.words()
            if len(wl) > 1000:  # sanity check the corpus actually downloaded
                words += [w.lower() for w in wl]
                dictionary_loaded = True
        except Exception:
            pass
        return Counter(words), dictionary_loaded

    WORD_FREQ, dict_loaded = build_norvig_model(DEFAULT_TEXT + " " + text_input)
    if not dict_loaded:
        st.warning(
            "⚠️ Full NLTK English word list failed to load — running on a "
            "small fallback vocabulary (just the words in your input text "
            "above). Corrections will be unreliable until this loads. "
            "Try re-running this cell/tab, or check your internet connection."
        )
    else:
        st.caption(f"Vocabulary loaded: {len(WORD_FREQ):,} unique words.")

    def edits1(word):
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
        inserts = [a + c + b for a, b in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(word):
        return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

    def correct(word):
        candidates = (
            ({word} & WORD_FREQ.keys())
            or (edits1(word) & WORD_FREQ.keys())
            or (edits2(word) & WORD_FREQ.keys())
            or {word}
        )
        return max(candidates, key=lambda w: WORD_FREQ[w])

    st.write(f"**Suggested correction:** `{correct(misspelled.lower())}`")

# ---------------------------------------------------------------------------
# Tab 3: Language Modeling (Weeks 2-3)
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Language Modeling")
    st.write("Bigram model (Laplace-smoothed, trained live) vs. pretrained GPT-2 perplexity.")

    from nltk.tokenize import sent_tokenize as sent_tok_lm

    def clean_tokens(sentence):
        return [t.lower() for t in word_tokenize(sentence) if t.isalpha()]

    sentences = sent_tok_lm(text_input)

    # Held-out evaluation: train the bigram model on one part of the text,
    # test perplexity on a DIFFERENT part it hasn't seen. Training and
    # testing on the same text (the earlier version of this tab) makes
    # perplexity look artificially low, since the model has literally
    # already seen every bigram it's being scored on — that's overfitting,
    # not a fair comparison against GPT-2, which has never seen this text.
    split_ok = len(sentences) >= 4
    if split_ok:
        cut = max(1, int(len(sentences) * 0.7))
        train_sents, test_sents = sentences[:cut], sentences[cut:]
        if not test_sents:  # guard tiny edge case
            train_sents, test_sents = sentences[:-1], sentences[-1:]
    else:
        train_sents, test_sents = sentences, sentences
        st.warning(
            "⚠️ Input text has fewer than 4 sentences, too short for a fair "
            "train/test split. Training and evaluating on the SAME text "
            "below — treat this perplexity as optimistic/overfit, not a "
            "fair comparison point. Add more sentences above for a proper "
            "held-out evaluation."
        )

    train_tokens = [w for s in train_sents for w in clean_tokens(s)]
    test_tokens = [w for s in test_sents for w in clean_tokens(s)]

    vocab = set(train_tokens)
    V = len(vocab)
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter(train_tokens)
    for w1, w2 in zip(train_tokens[:-1], train_tokens[1:]):
        bigram_counts[w1][w2] += 1

    def bigram_prob(w1, w2):
        num = bigram_counts[w1][w2] + 1
        den = unigram_counts.get(w1, 0) + V
        return num / den

    if split_ok:
        st.caption(
            f"Trained on {len(train_sents)} sentence(s) ({len(train_tokens)} "
            f"tokens); tested on {len(test_sents)} held-out sentence(s) "
            f"({len(test_tokens)} tokens) the model never saw during training."
        )

    test_bigrams = list(zip(test_tokens[:-1], test_tokens[1:]))
    if len(test_bigrams) >= 1:
        probs = [bigram_prob(w1, w2) for w1, w2 in test_bigrams]
        log_probs = [math.log(p) for p in probs]
        total_lp = sum(log_probs)
        n = len(test_bigrams)
        bigram_perplexity = math.exp(-total_lp / n)
        st.metric("Bigram model perplexity (Laplace-smoothed, held-out test text)",
                   f"{bigram_perplexity:.2f}")

        with st.expander("Show the calculation, step by step"):
            st.markdown(
                f"**Vocabulary size (V)** from training text: `{V}` unique words\n\n"
                "**Laplace-smoothed bigram probability formula:**\n\n"
                r"$$P(w_2 \mid w_1) = \dfrac{\text{count}(w_1, w_2) + 1}{\text{count}(w_1) + V}$$"
            )
            st.markdown("**Most frequent bigrams seen during training:**")
            top_bigrams = Counter(
                {(w1, w2): c for w1, counts in bigram_counts.items()
                 for w2, c in counts.items()}
            ).most_common(8)
            st.table([
                {"w1": w1, "w2": w2, "count(w1,w2)": c, "count(w1)": unigram_counts[w1]}
                for (w1, w2), c in top_bigrams
            ])

            st.markdown("**Per-bigram calculation on the held-out test text (first 10):**")
            rows = []
            running_total = 0.0
            for i, ((w1, w2), p, lp) in enumerate(zip(test_bigrams, probs, log_probs)):
                running_total += lp
                if i < 10:
                    rows.append({
                        "bigram": f"({w1}, {w2})",
                        "count(w1,w2)": bigram_counts[w1][w2],
                        "count(w1)": unigram_counts.get(w1, 0),
                        "P(w2|w1)": f"{p:.5f}",
                        "log P": f"{lp:.4f}",
                        "running Σ log P": f"{running_total:.4f}",
                    })
            st.table(rows)
            st.markdown(
                f"**Final perplexity** over all {n} held-out bigrams:\n\n"
                r"$$PP = \exp\left(-\frac{1}{N}\sum \log P(w_i \mid w_{i-1})\right)$$"
                f"\n\n$$PP = \\exp\\left(-\\frac{{{total_lp:.4f}}}{{{n}}}\\right) = {bigram_perplexity:.2f}$$"
            )
    else:
        st.info("Add more text above — need at least one bigram in the held-out portion.")

    if st.button("Compute GPT-2 perplexity (loads pretrained model, inference only)"):
        import torch
        tok, model = load_gpt2()
        enc = tok(text_input, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        gpt2_ppl = math.exp(out.loss.item())
        st.metric("GPT-2 perplexity on the full text (pretrained, never fine-tuned on it)",
                   f"{gpt2_ppl:.2f}")
        st.caption(
            "Lower perplexity = model assigns higher likelihood to the text. "
            "GPT-2's score reflects general-domain pretraining and no exposure "
            "to this specific text — a fair comparison against the bigram "
            "model's held-out (not overfit) perplexity above."
        )

# ---------------------------------------------------------------------------
# Tab 4: POS tagging (Week 3)
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("POS Tagging")
    from nltk import pos_tag
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("NLTK Averaged Perceptron (classical/statistical)")
        st.write(pos_tag(word_tokenize(text_input))[:30])
    with col2:
        st.subheader("spaCy pipeline (neural)")
        if st.button("Run spaCy POS tagging"):
            nlp = load_spacy()
            doc = nlp(text_input)
            st.write([(t.text, t.pos_) for t in doc][:30])

# ---------------------------------------------------------------------------
# Tab 5-6: Parsing (Weeks 5-6)
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Constituency & Dependency Parsing")
    st.write("spaCy dependency parse (pretrained neural parser).")
    if st.button("Run dependency parsing"):
        nlp = load_spacy()
        doc = nlp(text_input)
        rows = [(t.text, t.dep_, t.head.text) for t in doc]
        st.dataframe(rows, use_container_width=True)
        st.caption(
            "Full constituency parsing (e.g., via Stanza or a CFG parser) can be "
            "added as an extension — dependency parsing is shown here as the "
            "fast, pretrained default."
        )

# ---------------------------------------------------------------------------
# Tab 7: Distributional semantics (Week 7)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Distributional Semantics")
    st.write("Static embeddings (GloVe) vs. contextual embeddings (BERT).")

    @st.cache_resource(show_spinner="Downloading + loading GloVe vectors (50d)...")
    def load_glove():
        import gzip
        import io
        import urllib.request
        import numpy as np

        # Same underlying pretrained file gensim's downloader fetches
        # (Wikipedia 2014 + Gigaword 5, 400K vocab, 50d), loaded here
        # directly so the app doesn't depend on the gensim package.
        url = (
            "https://github.com/RaRe-Technologies/gensim-data/releases/"
            "download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
        )
        with urllib.request.urlopen(url) as resp:
            raw = resp.read()

        words, vecs = [], []
        with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
            next(f)  # header line: "<num_vectors> <dim>"
            for line in f:
                parts = line.rstrip().split(" ")
                words.append(parts[0])
                vecs.append(np.asarray(parts[1:], dtype=np.float32))

        matrix = np.vstack(vecs)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        unit = matrix / norms
        key_to_index = {w: i for i, w in enumerate(words)}
        return {"words": words, "unit": unit, "key_to_index": key_to_index}

    def glove_most_similar(glove, word, topn=5):
        import numpy as np
        idx = glove["key_to_index"][word]
        sims = glove["unit"] @ glove["unit"][idx]
        top_idx = np.argsort(-sims)
        results = []
        for i in top_idx:
            if i == idx:
                continue
            results.append((glove["words"][i], round(float(sims[i]), 4)))
            if len(results) >= topn:
                break
        return results

    if st.button("Load GloVe + visualize nearest neighbours"):
        glove = load_glove()
        all_text_tokens = [t.lower() for t in word_tokenize(text_input) if t.isalpha()]
        query_words = [w for w in dict.fromkeys(all_text_tokens) if w in glove["key_to_index"]][:8]
        for w in query_words:
            sims = glove_most_similar(glove, w, topn=5)
            st.write(f"**{w}** → {sims}")

    if st.button("Compare with BERT contextual embedding (same word, two contexts)"):
        import torch
        tok, model = load_bert()

        def get_embedding(sentence, word):
            enc = tok(sentence, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc)
            wid = tok.tokenize(word)[0]
            ids = tok.convert_ids_to_tokens(enc["input_ids"][0])
            idx = ids.index(wid) if wid in ids else 1
            return out.last_hidden_state[0, idx]

        import torch.nn.functional as F
        e1 = get_embedding("The bank raised interest rates.", "bank")
        e2 = get_embedding("He sat on the river bank.", "bank")
        sim = F.cosine_similarity(e1, e2, dim=0).item()
        st.metric("Cosine similarity of 'bank' across two different senses (BERT)", f"{sim:.3f}")
        st.caption(
            "Static embeddings (GloVe) give 'bank' one vector regardless of "
            "context; BERT's contextual embedding shifts with meaning — this "
            "is the core empirical contrast to discuss in your report."
        )

# ---------------------------------------------------------------------------
# Tab 8: Lexical semantics (Week 8)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Lexical Semantics (WordNet)")
    from nltk.corpus import wordnet as wn
    lookup = st.text_input("Look up a word in WordNet", "bank")
    synsets = wn.synsets(lookup)
    for s in synsets[:5]:
        st.write(f"**{s.name()}**: {s.definition()}")
        st.caption(f"Synonyms: {', '.join(l.name() for l in s.lemmas())}")

# ---------------------------------------------------------------------------
# Tab 9: Topic modeling (Week 9)
# ---------------------------------------------------------------------------
with tabs[7]:
    st.header("Topic Modeling")
    st.write("LDA (scikit-learn), trained live on the sentences of the input text.")
    from nltk.tokenize import sent_tokenize
    docs = [
        [w.lower() for w in word_tokenize(s) if w.isalpha() and w.lower() not in sw]
        for s in sent_tokenize(text_input)
    ]
    if len(docs) >= 2:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        doc_strings = [" ".join(d) for d in docs]
        vectorizer = CountVectorizer()
        dtm = vectorizer.fit_transform(doc_strings)
        feature_names = vectorizer.get_feature_names_out()

        num_topics = st.slider("Number of topics", 2, min(5, len(docs)), 2)
        lda = LatentDirichletAllocation(
            n_components=num_topics, max_iter=10, random_state=42
        )
        lda.fit(dtm)
        for i, topic in enumerate(lda.components_):
            top_terms = [feature_names[j] for j in topic.argsort()[-8:][::-1]]
            st.write(f"**Topic {i}:** " + ", ".join(top_terms))
    else:
        st.info("Add more sentences to the input text to fit a topic model.")

# ---------------------------------------------------------------------------
# Tab 10: NER / Information Extraction (Week 10)
# ---------------------------------------------------------------------------
with tabs[8]:
    st.header("Named Entity Recognition / Information Extraction")
    if st.button("Run spaCy NER"):
        nlp = load_spacy()
        doc = nlp(text_input)
        if doc.ents:
            st.dataframe([(e.text, e.label_) for e in doc.ents], use_container_width=True)
        else:
            st.info("No named entities found in this text — try a text with people, "
                    "organizations, or dates.")

# ---------------------------------------------------------------------------
# Tab 11: Summarization & Classification (Week 11)
# ---------------------------------------------------------------------------
with tabs[9]:
    st.header("Summarization & Classification")
    st.subheader("Extractive: TextRank (classical)")
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        parser = PlaintextParser.from_string(text_input, SumyTokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, 2)
        st.write(" ".join(str(s) for s in summary))
    except Exception as e:
        st.warning(f"sumy not available in this environment ({e}); install it per requirements.txt.")

    st.subheader("Abstractive: DistilBART-CNN (pretrained transformer)")
    if st.button("Run abstractive summarization"):
        import torch
        tok, model = load_summarizer()
        inputs = tok(text_input, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs, max_length=60, min_length=15,
                num_beams=4, do_sample=False,
            )
        summary_text = tok.decode(summary_ids[0], skip_special_tokens=True)
        st.write(summary_text)

# ---------------------------------------------------------------------------
# Tab 12: Sentiment Analysis (Week 12)
# ---------------------------------------------------------------------------
with tabs[10]:
    st.header("Sentiment Analysis & Opinion Mining")
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("VADER (lexicon-based, classical)")
        st.json(sia.polarity_scores(text_input))
    with col2:
        st.subheader("DistilBERT-SST2 (pretrained transformer)")
        if st.button("Run transformer sentiment"):
            sentiment_pipe = load_sentiment_pipeline()
            result = sentiment_pipe(text_input[:512])
            st.json(result)

st.divider()
st.caption(
    "Unified NLP Research Toolkit — built to accompany a comparative study of "
    "classical, statistical, and Transformer-based NLP across the core "
    "language-understanding pipeline. See README.md for the report structure "
    "and REPORT_OUTLINE.md for the write-up scaffold."
)