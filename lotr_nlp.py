"""Reusable LotR NLP utilities for preprocessing, topic modeling, and Word2Vec."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NLTK_RESOURCES = ["punkt", "wordnet", "omw-1.4", "stopwords"]


def ensure_nltk_resources(resources: Optional[Sequence[str]] = None) -> None:
    """Download required NLTK resources if they are missing."""
    resources = resources or NLTK_RESOURCES
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def load_text(path: str | Path, remove_newlines: bool = True) -> str:
    """Load a text file and optionally normalize newline characters."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        text = fh.read()
    return text.replace("\n", " ") if remove_newlines else text


def tokenize_text(text: str, preserve_case: bool = False, token_pattern: str = r"\w+") -> List[str]:
    """Tokenize raw text using a regular expression and optional casing control."""
    tokenizer = RegexpTokenizer(token_pattern)
    tokens = tokenizer.tokenize(text)
    return tokens if preserve_case else [token.lower() for token in tokens]


def get_stopwords(extra: Optional[Iterable[str]] = None) -> set[str]:
    """Return the default English stop words plus any extra terms."""
    ensure_nltk_resources(["stopwords"])
    words = set(nltk_stopwords.words("english"))
    if extra:
        words.update(word.lower() for word in extra)
    return words


def remove_stopwords(tokens: Sequence[str], stop_words: Optional[Iterable[str]] = None) -> List[str]:
    """Remove stop words from a token list."""
    stop_words = set(stop_words) if stop_words is not None else get_stopwords()
    return [token for token in tokens if token.lower() not in stop_words]


def lemmatize_tokens(tokens: Sequence[str]) -> List[str]:
    """Lemmatize a sequence of tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(
    text: str,
    preserve_case: bool = False,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> List[str]:
    """Convert raw text into cleaned, lemmatized tokens."""
    ensure_nltk_resources()
    tokens = tokenize_text(text, preserve_case=preserve_case)
    stop_words = get_stopwords(extra_stopwords)
    tokens = remove_stopwords(tokens, stop_words)
    return lemmatize_tokens(tokens)


def tokens_to_string(tokens: Sequence[str], separator: str = " ") -> str:
    """Join a token list into a single document string."""
    return separator.join(tokens)


def build_sentences(text: str, preserve_case: bool = False) -> List[List[str]]:
    """Tokenize text into cleaned sentences for Word2Vec or document chunks."""
    ensure_nltk_resources(["punkt"])
    sentences: List[List[str]] = []
    for raw_sentence in sent_tokenize(text):
        tokens = [token.lower() for token in word_tokenize(raw_sentence) if token.isalpha()]
        if tokens:
            sentences.append(tokens if preserve_case else [token.lower() for token in tokens])
    return sentences


def chunk_documents(sentences: Sequence[Sequence[str]], chunk_size: int = 20) -> List[str]:
    """Group sentences into document chunks for topic modeling."""
    chunks: List[str] = []
    for offset in range(0, len(sentences), chunk_size):
        current_chunk = sentences[offset : offset + chunk_size]
        flat_tokens = [token for sentence in current_chunk for token in sentence]
        if flat_tokens:
            chunks.append(tokens_to_string(flat_tokens))
    return chunks


class LemmaCountVectorizer(CountVectorizer):
    """A CountVectorizer that lemmatizes tokens during analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lemmatizer = WordNetLemmatizer()

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (self._lemmatizer.lemmatize(w) for w in analyzer(doc))


def build_document_term_matrix(
    documents: Sequence[str],
    max_df: float = 0.95,
    min_df: int = 2,
    stop_words: str | Iterable[str] = "english",
    lowercase: bool = True,
    token_pattern: str = r"\w+",
) -> tuple:
    """Build a document-term matrix from text documents."""
    vectorizer = LemmaCountVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words=stop_words,
        lowercase=lowercase,
        token_pattern=token_pattern,
        decode_error="ignore",
    )
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def print_top_words(model, feature_names: Sequence[str], n_top_words: int = 20) -> None:
    """Print top words for each topic in a topic model."""
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {' '.join(top_features)}")


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    sg: int = 0,
    workers: Optional[int] = None,
    epochs: int = 5,
) -> Word2Vec:
    """Train a Word2Vec model on a list of tokenized sentences."""
    workers = workers or max(1, os.cpu_count() or 1)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
    )
    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model


def fit_lda(matrix, n_components: int = 10, random_state: int = 2019, max_iter: int = 10) -> LatentDirichletAllocation:
    """Fit an LDA topic model."""
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=max_iter,
        learning_method="online",
        learning_offset=50.0,
        random_state=random_state,
    )
    lda.fit(matrix)
    return lda


def fit_nmf(matrix, n_components: int = 10, random_state: int = 2019, max_iter: int = 200) -> NMF:
    """Fit an NMF topic model."""
    nmf = NMF(n_components=n_components, random_state=random_state, max_iter=max_iter)
    nmf.fit(matrix)
    return nmf


def normalize_matrix(matrix):
    """Normalize a matrix of topic weights."""
    return normalize(matrix, norm="l1", axis=1)
