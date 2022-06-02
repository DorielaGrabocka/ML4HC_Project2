from typing import List

import gensim
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import long

from src.constants import PATH_FASTTEXT, PATH_WORD2VEC
from src.data_processing import PreprocessingOptions


class SentenceEmbedder:

    def __init__(self, embedding_model) -> None:
        """
        class to implement various ways of obtaining sentence embeddings using word2vec word vector representations
        """
        self.embedding_model = embedding_model

    def concatenate_word_vectors(self, x_preprocessed: List[List[str]], max_words: int) -> np.ndarray:
        """
        Returns array of n_sentences x max_words x word_vector_size
        shorter sequences are zero padded
        """
        output = np.zeros((len(x_preprocessed), max_words, self.embedding_model.vectors.shape[1]), dtype=np.float16)
        for idx, sentence in enumerate(x_preprocessed):
            vectors = self._compute_vectors_for_sentence(sentence)
            # if there is a longer sequence, cut it off
            vectors = vectors[:max_words]
            output[idx, 0:len(vectors)] = np.stack(vectors).astype(np.float16)

        assert len(output) == len(x_preprocessed)
        return output

    def sum_word_vectors(self, x_preprocessed: List[List[str]]) -> np.ndarray:
        """
        Returns array of n_sentences x word_vector_size
        """
        output = np.zeros((len(x_preprocessed), self.embedding_model.vectors.shape[1]), dtype=np.float32)
        for idx, sentence in enumerate(x_preprocessed):
            vectors = self._compute_vectors_for_sentence(sentence)
            output[idx] = np.sum(vectors, axis=0).astype(np.float32)

        assert len(output) == len(x_preprocessed)
        return output

    def _compute_vectors_for_sentence(self, sentence: List[str]) -> np.ndarray:
        vectors = [self.embedding_model[word] for word in sentence if word in self.embedding_model]
        if len(vectors) == 0:
            # there are some degenerate sentences eg: BACKGROUND	http://www.clinicaltrials.gov .
            vectors = [np.zeros(self.embedding_model.vectors.shape[1])]
        return vectors

    def print_unknown_words_percentage(self, x_preprocessed: List[List[str]]) -> None:
        total = 0
        unknown = 0
        for sentence in x_preprocessed:
            total += len(sentence)
            for word in sentence:
                if word not in self.embedding_model:
                    unknown += 1

        print(f"Unknown percentage {unknown/total * 100}")

    def compute_longest_sentence_length(self, x_preprocessed: List[List[str]]) -> int:
        lengths = [len(sentence) for sentence in x_preprocessed]
        print(f"Relevant lenght statistics: Mean {np.mean(lengths)} Std {np.std(lengths)} Max {max(lengths)} Min {min(lengths)}")
        return max(lengths)


def get_embedding_config(preprocessing_options: PreprocessingOptions, embedding_version: str, vector_size: int, max_words: int):
    return preprocessing_options.get_current_options() + "_" + embedding_version + "_" + str(vector_size) + "_" + str(max_words)


def train_and_save_embedding_model(x_preprocessed: List[List[str]], sg: int, vector_size: int, embedding_type: str = "word2vec"):

    if embedding_type == "word2vec":
        if sg == 0:
            print("Training cbow model of word2vec...")
            model = Word2Vec(x_preprocessed, vector_size=vector_size, sg=sg, min_count=5, workers=1, seed=0)
            model.save(PATH_WORD2VEC + "cbow" + "_" + str(vector_size))
        elif sg == 1:
            print("Training Skip_N-gram model of word2vec...")
            model = Word2Vec(x_preprocessed, vector_size=vector_size, sg=sg, min_count=5, workers=1, seed=0)
            model.save(PATH_WORD2VEC + "Skip_N-gram" + "_" + str(vector_size))
        else:
            raise ValueError("Invalid sg")

    else:  # fasttext embedding
        if sg == 0:
            print("Training cbow mode of fasttext...")
            model = FastText(x_preprocessed, vector_size=vector_size, sg=sg, min_count=5, workers=1, seed=0)
            model.save(PATH_FASTTEXT + "cbow" + "_" + str(vector_size))
        elif sg == 1:
            print("Training Skip_N-gram model of fasttext...")
            model = FastText(x_preprocessed, vector_size=vector_size, sg=sg, min_count=5, workers=1, seed=0)
            model.save(PATH_FASTTEXT + "Skip_N-gram" + "_" + str(vector_size))
        else:
            raise ValueError("Invalid sg")


def load_embedding(version: str, vector_size: int, embedding_type: str = "word2vec"):
    if embedding_type == "word2vec":
        return KeyedVectors.load(PATH_WORD2VEC + version + "_" + str(vector_size), mmap='r').wv
    else:
        return FastText.load(PATH_FASTTEXT + version + "_" + str(vector_size)).wv


def fit_tf_idf(x_preprocessed):
    """
    Learn vocabulary and idf from given data
    """
    # inputs are already tokenized, see https://stackoverflow.com/questions/48671270/use-sklearn-tfidfvectorizer-with-already-tokenized-inputs
    def identity_tokenizer(text):
        return text

    # already lowercase as well
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    tfidf.fit(x_preprocessed)

    return tfidf
