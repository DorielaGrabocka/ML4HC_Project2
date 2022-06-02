import tensorflow as tf
import tensorflow_addons as tfa


def compile_tf_model(model, from_logits: bool = True):
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=5, average="macro", name="macro_f1_score"),
                  tfa.metrics.F1Score(num_classes=5, average="weighted", name="weighted_f1_score")])
