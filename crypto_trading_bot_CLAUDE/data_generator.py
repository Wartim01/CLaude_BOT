import tensorflow as tf
import pandas as pd
import os

def csv_data_generator(file_path, sequence_length=60):
    # Charger le fichier CSV par chunks pour limiter l'utilisation de la mémoire
    for chunk in pd.read_csv(file_path, chunksize=1000):
        data = chunk.values.astype('float32')
        num_sequences = len(data) - sequence_length + 1
        for i in range(num_sequences):
            yield data[i:i+sequence_length]

def create_dataset(file_path, sequence_length=60, batch_size=32):
    # Créer un tf.data.Dataset à partir du générateur
    dataset = tf.data.Dataset.from_generator(
        lambda: csv_data_generator(file_path, sequence_length),
        output_signature=tf.TensorSpec(shape=(sequence_length, None), dtype=tf.float32)
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
