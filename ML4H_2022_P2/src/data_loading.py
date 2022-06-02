import dataclasses
import pickle
from typing import List, Tuple

import numpy as np

from src.constants import (PATH_DEV_DATA, PATH_EMBEDDINGS_DATA_WORD2VEC, PATH_EMBEDDINGS_DATA_FASTTEXT, PATH_LABELS,
                           PATH_PREPROCESSED_DATA, PATH_TEST_DATA,
                           PATH_TRAIN_DATA)
from src.data_processing import PreprocessingOptions
from src.embeddings import get_embedding_config

def persist_embeddings(train_embeddings: np.ndarray, dev_embeddings: np.ndarray, test_embeddings: np.ndarray,
                       preprocessing_options: PreprocessingOptions, embedding_version: str, vector_size: int, max_words: int,
                       mode: str, embedding_type : str = "word2vec"):
    # the embeddings depend both on the preprocessing options and word2vec/fasttext model settings
    #current_config = get_embedding_config(preprocessing_options, embedding_version, vector_size, max_words)
    if embedding_type == "word2vec":
      print("Persisting word2vec "+ embedding_version+"...")
      current_config = get_embedding_config(preprocessing_options, embedding_version, vector_size, max_words)
    else:
      print("Persisting fasttext "+ embedding_version+"...")
      current_config = get_embedding_config(preprocessing_options, embedding_version, vector_size, max_words)

    path_to_embedding = get_path_of_embedding(embedding_type)
    # save embeddings for the current config
    np.save(path_to_embedding + "train_" + current_config + "_" + mode, train_embeddings)
    np.save(path_to_embedding + "dev_" + current_config + "_" + mode, dev_embeddings)
    np.save(path_to_embedding + "test_" + current_config + "_" + mode, test_embeddings)


def load_embeddings(preprocessing_options: PreprocessingOptions, embedding_version: str, vector_size: int, max_words: int,
                    embedding_type : str = "word2vec", mode: str = "concatenation"):
    if embedding_type == "word2vec":
      print("Loading word2vec...")
      current_config = get_embedding_config(preprocessing_options, embedding_version, vector_size, max_words)
    else:
      print("Loading fasttext...")
      current_config = get_embedding_config(preprocessing_options, embedding_version, vector_size, max_words)
    
    path_to_embedding = get_path_of_embedding(embedding_type)
    
    train_embeddings = np.load(path_to_embedding + "train_" + current_config + "_" + mode + ".npy")
    dev_embeddings = np.load(path_to_embedding + "dev_" + current_config + "_" + mode + ".npy")
    test_embeddings = np.load(path_to_embedding + "test_" + current_config + "_" + mode + ".npy")
    print("Loading done from: "+path_to_embedding)
    return train_embeddings, dev_embeddings, test_embeddings

def get_path_of_embedding(embedding_type: str):
  """Method to distinguish between different word embeddings in order to 
     load them from their specified file paths"""
  if embedding_type == "fasttext":
        path_to_embedding = PATH_EMBEDDINGS_DATA_FASTTEXT
  else:
      path_to_embedding = PATH_EMBEDDINGS_DATA_WORD2VEC
  return path_to_embedding

def persist_labels(y_train: np.ndarray, y_dev: np.ndarray, y_test: np.ndarray) -> None:
    np.save(PATH_LABELS + "train", y_train)
    np.save(PATH_LABELS + "dev", y_dev)
    np.save(PATH_LABELS + "test", y_test)


def load_labels() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.load(PATH_LABELS + "train" + ".npy")
    y_dev = np.load(PATH_LABELS + "dev" + ".npy")
    y_test = np.load(PATH_LABELS + "test" + ".npy")

    return y_train, y_dev, y_test


def persist_preprocessed_data(x: List[List[str]], preprocessing_options: PreprocessingOptions, dataset_type: str) -> None:
    path_x = _get_path(preprocessing_options, dataset_type)

    with open(path_x, "wb") as fp:
        pickle.dump(x, fp)


def load_preprocessed_data(preprocessing_options: PreprocessingOptions, dataset_type: str) -> List[List[str]]:
    path_x = _get_path(preprocessing_options, dataset_type)

    with open(path_x, "rb") as fp:   # Unpickling
        x = pickle.load(fp)

    return x


def _get_path(preprocessing_options: PreprocessingOptions, dataset_type: str) -> str:
    current_options = preprocessing_options.get_current_options()
    return PATH_PREPROCESSED_DATA + "x_" + dataset_type + "_" + current_options


def load_raw_datasets() -> Tuple[List[str], List[str], List[str]]:
    train = load_data(PATH_TRAIN_DATA)
    dev = load_data(PATH_DEV_DATA)
    test = load_data(PATH_TEST_DATA)

    return train, dev, test


def load_data(path: str) -> List[str]:
    with open(path) as f:
        lines = f.readlines()

    return lines
