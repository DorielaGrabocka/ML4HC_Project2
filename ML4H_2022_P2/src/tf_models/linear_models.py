import tensorflow as tf
from src.tf_models.utils import compile_tf_model


def get_simple_linear_classifier(input_features: int):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_features,)),
        tf.keras.layers.Dense(5, activation="sigmoid")
    ])
    compile_tf_model(model, from_logits=False)

    return model
