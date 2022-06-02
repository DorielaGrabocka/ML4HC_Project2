import multiprocessing
import os
import pickle
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from src.constants import *
from src.runtime_benchmarking import timing


@dataclass
class PreprocessingOptions:
    # be ready to run out of memory if too many processes are used
    n_processes: int = 4
    remove_punctuation: bool = True
    remove_stop_words: bool = False
    lemmatisation: bool = False

    def get_current_options(self) -> str:
        current_options = ""
        if self.remove_punctuation:
            current_options += "remove_punctuation"
        if self.remove_stop_words:
            current_options += "_remove_stop_words"
        if self.lemmatisation:
            current_options += "_lemmatisation"
        return current_options


def preprocess_raw_datasets(train: List[str], dev: List[str], test: List[str], preprocessing_options: PreprocessingOptions):
    print("Preprocessing train data")
    x_preprocessed_train, y_train = _preprocess_raw_dataset(train, preprocessing_options)
    print("Preprocessing dev data")
    x_preprocessed_dev, y_dev = _preprocess_raw_dataset(dev, preprocessing_options)
    print("Preprocessing test data")
    x_preprocessed_test, y_test = _preprocess_raw_dataset(test, preprocessing_options)

    return x_preprocessed_train, y_train, x_preprocessed_dev, y_dev, x_preprocessed_test, y_test


@timing
def _preprocess_raw_dataset(raw_dataset: List[str], preprocessing_options: PreprocessingOptions) -> Tuple[List[List[str]], np.ndarray]:
    x_raw, y_raw = get_raw_x_y(raw_dataset)
    x = get_x_token(x_raw, preprocessing_options)
    y = encode_labels(y_raw)
    assert len(x) == len(y)
    return x, y


def get_x_token(x_raw: List[str], preprocessing_options: PreprocessingOptions) -> List[List[str]]:
    """
    converts each sentence (string) in the dataset into a list of tokens
    done on multiple processes because the dataset is quite large and would otherwise take a lot of time
    """
    n_processes = preprocessing_options.n_processes
    # split available data for each thread equally
    n_sentences = len(x_raw)
    sentences_per_thread = n_sentences // n_processes

    def _tokenize_sentences(start_idx: int, end_idx: int) -> None:
        print(f"Process {os.getpid()} is running on elements from {start_idx} to {end_idx}")
        current_sentences = x_raw[start_idx: end_idx]
        current_tokens = [get_sentence_tokens(sentence, preprocessing_options) for sentence in current_sentences]
        with open(TMP_FILE_LOC + str(start_idx), "wb") as fp:
            pickle.dump(current_tokens, fp)

    partial_results_idxs = []
    cur_idx = 0
    processes = []
    for process_nr in range(1, n_processes+1):
        partial_results_idxs.append(cur_idx)
        end_idx = cur_idx + sentences_per_thread
        if process_nr == n_processes:
            # make sure we cover everything
            end_idx = n_sentences

        proc = multiprocessing.Process(target=_tokenize_sentences, args=(cur_idx, end_idx))
        proc.start()
        processes.append(proc)

        cur_idx = end_idx

    for proc in processes:
        proc.join()

    x_tokens = []
    # load back partial results
    for idx in partial_results_idxs:
        with open(TMP_FILE_LOC + str(idx), "rb") as fp:   # Unpickling
            x = pickle.load(fp)
        x_tokens.extend(x)

    assert all([x is not None for x in x_tokens])

    return x_tokens


def encode_labels(y_raw: List[str]) -> np.ndarray:
    mapping = {
        "BACKGROUND": 0,
        "OBJECTIVE": 1,
        "METHODS": 2,
        "RESULTS": 3,
        "CONCLUSIONS": 4
    }
    y = []
    for label in y_raw:
        y.append(mapping[label])
    return np.array(y)


def encode_one_hot_labels(y_labels: np.ndarray) -> np.ndarray:
    y = np.zeros((len(y_labels), 5))
    for i, label in enumerate(y_labels):
        encoding = np.zeros(5)
        encoding[label] = 1
        y[i] = encoding
    return y


def get_sentence_tokens(raw_sentence: str, preprocessing_options: PreprocessingOptions) -> List[str]:
    # there is always an \n -> remove that
    raw_sentence = raw_sentence.strip()

    if preprocessing_options.remove_punctuation:
        raw_sentence = _remove_punctuation(raw_sentence)

    tokens = raw_sentence.lower().split(" ")
    # remove empty elements -> arise because punctuation removal renders two spaces next to each other
    tokens = [token for token in tokens if token != '']

    if preprocessing_options.remove_stop_words:
        tokens = _remove_stop_words(tokens)

    if preprocessing_options.lemmatisation:
        tokens = _lemmatize(tokens)

    return tokens


def _lemmatize(tokens: List[str]) -> List[str]:
    # idea taken from https://stackoverflow.com/a/15590384
    lemmatizer = WordNetLemmatizer()
    raw_pos_tags = nltk.pos_tag(tokens)
    pos_tags = [_get_wordnet_pos(x[1]) for x in raw_pos_tags]
    return [lemmatizer.lemmatize(word, pos) for word, pos in zip(tokens, pos_tags)]


def _get_wordnet_pos(tag: str):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def _remove_stop_words(tokens: List[str]) -> List[str]:
    stopwords_set = set(stopwords.words('english'))
    return [word for word in tokens if word not in stopwords_set]


def _remove_punctuation(s: str) -> str:
    """
    Method based on accepted answer here: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    Adapted NOT to remove % -> Might be indicative of RESULTS part
    """
    return s.translate(str.maketrans('', '', string.punctuation.replace('%', '')))


def get_raw_x_y(lines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Example return:
    x =
    ['All subjects were oriented to 6-minute walk tests , use of bronchodilators was controlled , and standard encouragement was given during each walk test .\n',
    'Outcome measures were the distance walked in 6 minutes , change in oxyhemoglobin saturation during the walk , and breathlessness using a modified Borg Scale .\n',
    'The use of a wheeled walker resulted in a significant increase in 6-minute walking distance , a significant reduction in hypoxemia with walking and a significant reduction in breathlessness during the walk test .\n',
    'The use of a wheeled walker resulted in significant decreases in disability , hypoxemia , and breathlessness during a 6-minute walk test .\n',
    'By reducing disability and breathlessness , a wheeled walker may improve quality of life in individuals with severe impairment in lung function .\n']

    y =
    ['METHODS', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'CONCLUSIONS']
    """
    x_raw, y_raw = [], []
    for line in lines:
        if _is_labelled_sentence(line):
            y, x = line.split("\t")
            x_raw.append(x)
            y_raw.append(y)
    return x_raw, y_raw


def _is_labelled_sentence(line: str) -> bool:
    """
    checks if the given string represents a (sentence, label) data point
    those have the form: LABEL\t...
    """
    ans = False
    for label in LABELS:
        if line.startswith(label):
            ans = True
    return ans


def get_value_from_tuple(values_to_split):
    splitted_result = []
    for v in values_to_split:
        splitted_result.append(v[0])
    return splitted_result
