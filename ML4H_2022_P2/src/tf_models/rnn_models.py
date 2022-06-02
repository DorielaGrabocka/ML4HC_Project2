import tensorflow as tf
from src.tf_models.utils import compile_tf_model


def rnn_model_factory(model_name: str):
    if model_name == "large_bidirectional_lstm":
        model = get_large_bidirectional_lstm()
    if model_name == "bidirectional_lstm":
        model = get_bidirectional_lstm()
    elif model_name == "lstm":
        model = get_lstm()
    elif model_name == "rnn":
        model = get_rnn()
    else:
        ValueError(f"Unknown model {model_name}")

    compile_tf_model(model)

    return model


def get_large_bidirectional_lstm() -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model


def get_bidirectional_lstm() -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model


def get_lstm() -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model


def get_rnn() -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(32, return_sequences=True),
        tf.keras.layers.SimpleRNN(16),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)
    ])
    return model


def load_trained_model(path: str):
    return tf.keras.models.load_model(path)
