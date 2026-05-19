"""Example runner for LotR NLP preprocessing, topic modeling, and Word2Vec."""
from __future__ import annotations

import argparse
from pathlib import Path

from lotr_nlp import (
    build_document_term_matrix,
    build_sentences,
    chunk_documents,
    fit_lda,
    fit_nmf,
    load_text,
    print_top_words,
    preprocess_text,
    train_word2vec,
    tokens_to_string,
)


def run_topic_modeling(text_path: Path, n_topics: int = 10) -> None:
    raw_text = load_text(text_path)
    sentences = build_sentences(raw_text)
    documents = chunk_documents(sentences, chunk_size=40)

    tf_matrix, vectorizer = build_document_term_matrix(documents)
    feature_names = vectorizer.get_feature_names_out()

    print("\nFitting LDA model...\n")
    lda = fit_lda(tf_matrix, n_components=n_topics)
    print_top_words(lda, feature_names, n_top_words=15)

    print("\nFitting NMF model...\n")
    nmf = fit_nmf(tf_matrix, n_components=n_topics)
    print_top_words(nmf, feature_names, n_top_words=15)


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
    parser.add_argument("--text", type=Path, default=Path("the_lord_of_the_rings_text/Lord_of_the_Rings_complete.txt"), help="Path to the LotR text file.")
    parser.add_argument("--topics", type=int, default=10, help="Number of topics for LDA/NMF.")
    parser.add_argument("--word2vec", action="store_true", help="Run Word2Vec analysis.")
    args = parser.parse_args()

    run_topic_modeling(args.text, args.topics)

    if args.word2vec:
        run_word2vec(args.text)


if __name__ == "__main__":
    main()
