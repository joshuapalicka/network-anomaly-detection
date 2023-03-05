import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

"""
train_data = pd.read_csv('eda_simple_classification/network_data_mod_train.csv')
test_data = pd.read_csv('eda_simple_classification/network_data_mod_test.csv')

frames = [train_data, test_data]

dataframe  = pd.concat(frames)
raw_data = dataframe.values
#dataframe.head()
#raw_data

# The last element contains the labels
labels = raw_data[:, -1]

#print(test_data.columns )

data = raw_data[:, 0:-1]
#print(data)

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=34
)

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
"""

class AnomalyDetector(Model):
  def __init__(self, lastLayer):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(lastLayer, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


#print(decoded_data)


