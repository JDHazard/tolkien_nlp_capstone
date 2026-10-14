"""Example runner for LotR NLP preprocessing, topic modeling, and Word2Vec."""
from __future__ import annotations

import argparse
from pathlib import Path

from lotr_nlp import (
    build_book_chapter_documents,
    build_document_term_matrix,
    build_sentences,
    build_tfidf_matrix,
    compute_coherence,
    fit_lda,
    fit_nmf,
    fit_embedding_topic_model,
    get_cluster_topic_terms,
    get_topic_term_lists,
    load_text,
    plot_topic_trends,
    print_top_words,
    train_word2vec,
)


def run_chapter_level_comparison(book_dir: Path, n_topics: int = 10, n_top_words: int = 15) -> None:
    chapters = build_book_chapter_documents(
        book_dir,
        preserve_case=False,
        extra_stopwords=["said", "come", "came"],
    )
    if not chapters:
        print(f"No chapter documents found in {book_dir}")
        return

    documents = [chapter["document"] for chapter in chapters]
    tokenized_documents = [chapter["tokens"] for chapter in chapters]

    print("\nChapter-level LDA (books combined):\n")
    lda_matrix, lda_vectorizer = build_document_term_matrix(documents)
    lda = fit_lda(lda_matrix, n_components=n_topics)
    lda_terms = get_topic_term_lists(lda, lda_vectorizer.get_feature_names_out(), n_top_words)
    print_top_words(lda, lda_vectorizer.get_feature_names_out(), n_top_words)
    print(f"LDA coherence (c_v): {compute_coherence(lda_terms, tokenized_documents):.4f}\n")
    lda_plot = plot_topic_trends(lda.transform(lda_matrix), chapters, top_n=5, title="Chapter-level LDA Topic Trends")
    lda_plot.savefig("lda_topic_trends.png")
    print("Saved LDA topic trend plot to lda_topic_trends.png")

    print("\nTF-IDF + NMF (chapter-level):\n")
    nmf_matrix, nmf_vectorizer = build_tfidf_matrix(documents)
    nmf = fit_nmf(nmf_matrix, n_components=n_topics)
    nmf_terms = get_topic_term_lists(nmf, nmf_vectorizer.get_feature_names_out(), n_top_words)
    print_top_words(nmf, nmf_vectorizer.get_feature_names_out(), n_top_words)
    print(f"NMF coherence (c_v): {compute_coherence(nmf_terms, tokenized_documents):.4f}\n")
    nmf_plot = plot_topic_trends(nmf.transform(nmf_matrix), chapters, top_n=5, title="TF-IDF + NMF Topic Trends")
    nmf_plot.savefig("nmf_topic_trends.png")
    print("Saved NMF topic trend plot to nmf_topic_trends.png")

    print("\nEmbedding-based topic modeling (KMeans on sentence embeddings):\n")
    labels, embeddings, _ = fit_embedding_topic_model(documents, n_clusters=n_topics)
    cluster_terms = get_cluster_topic_terms(documents, labels, n_top_words)
    for label, terms in cluster_terms:
        print(f"Cluster {label}: {' '.join(terms)}")
    print(f"Embedding cluster coherence (c_v): {compute_coherence([terms for _, terms in cluster_terms], tokenized_documents):.4f}\n")


def run_word2vec(text_path: Path) -> None:
    raw_text = load_text(text_path)
    sentences = build_sentences(raw_text)
    model_cbow = train_word2vec(sentences, sg=0)
    model_skipgram = train_word2vec(sentences, sg=1)

    print("\nMost similar words for 'ring' (CBOW):")
    print(model_cbow.wv.most_similar("ring", topn=10))
    print("\nMost similar words for 'ring' (SkipGram):")
    print(model_skipgram.wv.most_similar("ring", topn=10))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LotR NLP pipelines.")
    parser.add_argument(
        "--books",
        type=Path,
        default=Path("the_lord_of_the_rings_text"),
        help="Directory with the trilogy book text files.",
    )
    parser.add_argument("--topics", type=int, default=10, help="Number of topics for the comparison models.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run chapter-level LDA, TF-IDF+NMF, and embedding-based topic modeling comparisons.",
    )
    parser.add_argument("--word2vec", action="store_true", help="Run Word2Vec analysis.")
    args = parser.parse_args()

    if args.compare:
        run_chapter_level_comparison(args.books, args.topics)
    if args.word2vec:
        run_word2vec(args.books / "Lord_of_the_Rings_complete.txt")


if __name__ == "__main__":
    main()
