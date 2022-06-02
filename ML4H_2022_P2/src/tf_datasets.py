import tensorflow as tf


def create_tf_datasets(x_embeddings_train, y_train, x_embeddings_dev, y_dev, x_embeddings_test, y_test):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_embeddings_train, y_train))
    dev_dataset = tf.data.Dataset.from_tensor_slices((x_embeddings_dev, y_dev))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_embeddings_test, y_test))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 10000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    dev_dataset = dev_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, dev_dataset, test_dataset
