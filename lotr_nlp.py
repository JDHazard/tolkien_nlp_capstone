"""Reusable LotR NLP utilities for preprocessing, topic modeling, and Word2Vec."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, Word2Vec
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESOURCE_PATHS = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "stopwords": "corpora/stopwords",
}

DEFAULT_CHAPTER_PATTERN = re.compile(
    r"^(?:\* PROLOGUE \*|BOOK\s+[IVX]+|Chapter\s+\d+\.?|\d+\..+)$",
    re.IGNORECASE,
)


def ensure_nltk_resources(resources: Optional[Sequence[str]] = None) -> None:
    """Download required NLTK resources if they are missing."""
    resources = resources or list(RESOURCE_PATHS)
    for resource in resources:
        resource_path = RESOURCE_PATHS.get(resource, f"corpora/{resource}")
        try:
            nltk.data.find(resource_path)
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


class LemmaTfidfVectorizer(TfidfVectorizer):
    """A TfidfVectorizer that lemmatizes tokens during analysis."""

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
) -> tuple[Any, LemmaCountVectorizer]:
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


def build_tfidf_matrix(
    documents: Sequence[str],
    max_df: float = 0.95,
    min_df: int = 2,
    stop_words: str | Iterable[str] = "english",
    lowercase: bool = True,
    token_pattern: str = r"\w+",
) -> tuple[Any, LemmaTfidfVectorizer]:
    """Build a TF-IDF matrix from text documents."""
    vectorizer = LemmaTfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        stop_words=stop_words,
        lowercase=lowercase,
        token_pattern=token_pattern,
        decode_error="ignore",
    )
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def get_topic_term_lists(model: Any, feature_names: Sequence[str], n_top_words: int = 20) -> List[List[str]]:
    """Return top token lists for each topic in a topic model."""
    return [
        [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        for topic in model.components_
    ]


def print_top_words(model: Any, feature_names: Sequence[str], n_top_words: int = 20) -> None:
    """Print top words for each topic in a topic model."""
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx + 1}: {' '.join(top_features)}")


def compute_coherence(
    topics: List[List[str]],
    tokenized_texts: Sequence[Sequence[str]],
    coherence: str = "c_v",
) -> float:
    """Compute coherence score for a list of topic terms."""
    dictionary = Dictionary(tokenized_texts)
    return CoherenceModel(
        topics=topics,
        texts=list(tokenized_texts),
        dictionary=dictionary,
        coherence=coherence,
    ).get_coherence()


def embed_documents(documents: Sequence[str], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Any, Any]:
    """Embed documents using a SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding-based topic modeling. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=False, convert_to_numpy=True)
    return embeddings, model


def fit_embedding_topic_model(
    documents: Sequence[str],
    n_clusters: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    random_state: int = 2019,
) -> Tuple[List[int], Any, Any]:
    """Cluster documents by semantic embeddings to produce topics."""
    embeddings, embed_model = embed_documents(documents, model_name)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.tolist(), embeddings, embed_model


def get_cluster_topic_terms(
    documents: Sequence[str],
    labels: Sequence[int],
    n_top_words: int = 15,
    stop_words: str | Iterable[str] = "english",
) -> List[Tuple[int, List[str]]]:
    """Extract cluster-level topic terms using TF-IDF on clustered documents."""
    tfidf_matrix, vectorizer = build_tfidf_matrix(documents, stop_words=stop_words)
    feature_names = vectorizer.get_feature_names_out()
    clusters: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    cluster_terms: List[Tuple[int, List[str]]] = []
    for label, indices in sorted(clusters.items()):
        if not indices:
            continue
        submatrix = tfidf_matrix[indices]
        mean_scores = submatrix.mean(axis=0).A1
        top_indices = mean_scores.argsort()[::-1][:n_top_words]
        cluster_terms.append((label, [feature_names[i] for i in top_indices]))
    return cluster_terms


def split_text_into_chapters(text: str, title_pattern: re.Pattern = DEFAULT_CHAPTER_PATTERN) -> List[Tuple[str, str]]:
    """Split a long text into chapter segments using heading patterns."""
    lines = text.splitlines()
    chapters: List[Tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and title_pattern.match(stripped):
            if current_title and current_lines:
                chapters.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title and current_lines:
        chapters.append((current_title, "\n".join(current_lines).strip()))
    return chapters


def load_book_texts(book_dir: str | Path) -> List[Tuple[str, str]]:
    """Load the trilogy book files from a directory."""
    book_dir = Path(book_dir)
    candidates = sorted(book_dir.glob("the_*.txt"))
    book_files = [path for path in candidates if path.name != "Lord_of_the_Rings_complete.txt"]
    return [
        (path.stem.replace("the_", "").replace("_", " ").title(), load_text(path))
        for path in book_files
    ]


def build_book_chapter_documents(
    book_dir: str | Path,
    preserve_case: bool = False,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> List[dict[str, Any]]:
    """Return chapter-level documents for each book in the trilogy."""
    chapter_documents: List[dict[str, Any]] = []
    for book_name, text in load_book_texts(book_dir):
        for chapter_index, (chapter_title, chapter_text) in enumerate(split_text_into_chapters(text), start=1):
            if not chapter_text.strip():
                continue
            tokenized = preprocess_text(chapter_text, preserve_case=preserve_case, extra_stopwords=extra_stopwords)
            chapter_documents.append(
                {
                    "book": book_name,
                    "chapter": chapter_index,
                    "title": chapter_title,
                    "text": chapter_text,
                    "tokens": tokenized,
                    "document": tokens_to_string(tokenized),
                }
            )
    return chapter_documents


def build_topic_timeline_dataframe(
    topic_weights: Any,
    chapter_documents: Sequence[dict[str, Any]],
    label_prefix: str = "Topic",
) -> pd.DataFrame:
    """Build a tidy DataFrame for plotting topic prevalence across chapters."""
    df = pd.DataFrame(
        topic_weights,
        columns=[f"{label_prefix} {i + 1}" for i in range(topic_weights.shape[1])],
    )
    df["book"] = [chapter["book"] for chapter in chapter_documents]
    df["chapter"] = [chapter["chapter"] for chapter in chapter_documents]
    df["position"] = range(1, len(chapter_documents) + 1)
    return df


def plot_topic_trends(
    topic_weights: Any,
    chapter_documents: Sequence[dict[str, Any]],
    top_n: int = 5,
    title: str = "Topic Trends Across Trilogy",
) -> plt.Figure:
    """Plot topic prevalence across the trilogy timeline."""
    df = build_topic_timeline_dataframe(topic_weights, chapter_documents)
    topic_columns = [col for col in df.columns if col.startswith("Topic")]
    average_strength = df[topic_columns].mean().sort_values(ascending=False)
    top_topics = average_strength.head(top_n).index.tolist()
    plot_df = df[["position", "book", "chapter"] + top_topics].melt(
        id_vars=["position", "book", "chapter"],
        value_vars=top_topics,
        var_name="topic",
        value_name="weight",
    )
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=plot_df, x="position", y="weight", hue="topic", marker="o")
    boundaries = []
    last_book = plot_df.iloc[0]["book"]
    for idx, book in enumerate(df["book"], start=1):
        if book != last_book:
            boundaries.append(idx - 0.5)
            last_book = book
    for boundary in boundaries:
        plt.axvline(boundary, color="gray", linestyle="--", alpha=0.4)
    plt.title(title)
    plt.xlabel("Chapter position across trilogy")
    plt.ylabel("Topic strength")
    plt.tight_layout()
    return plt.gcf()


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


def fit_lda(matrix: Any, n_components: int = 10, random_state: int = 2019, max_iter: int = 10) -> LatentDirichletAllocation:
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


def fit_nmf(matrix: Any, n_components: int = 10, random_state: int = 2019, max_iter: int = 200) -> NMF:
    """Fit an NMF topic model."""
    nmf = NMF(n_components=n_components, random_state=random_state, max_iter=max_iter)
    nmf.fit(matrix)
    return nmf


def normalize_matrix(matrix: Any) -> Any:
    """Normalize a matrix of topic weights."""
    return normalize(matrix, norm="l1", axis=1)
