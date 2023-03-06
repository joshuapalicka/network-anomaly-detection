import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class AnomalyDetector(Model):
  def __init__(self, lastLayer):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(35, activation="relu"),
      layers.Dense(30, activation="relu"),
      layers.Dense(25, activation="relu"),
      layers.Dense(20, activation="relu"),
      layers.Dense(15, activation="relu"),
      layers.Dense(10, activation="relu")
      
      ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(10, activation="relu"),
      layers.Dense(15, activation="relu"),
      layers.Dense(20, activation="relu"),
      layers.Dense(25, activation="relu"),
      layers.Dense(30, activation="relu"),
      layers.Dense(35, activation="relu"),
      layers.Dense(lastLayer, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
    
    def test(self):
        print("Iam being called")

    def please():
        print("called")
