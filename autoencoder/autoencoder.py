from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.losses import mae
from tensorflow.math import less
from tensorflow.keras.models import Model
import numpy as np


class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = Sequential([
            layers.Dense(32, activation="relu"),  # zc
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])

        self.decoder = Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(46, activation="sigmoid")])  # zed

    def call(self, x):
        encoded = self.encoder(x)
        #print(encoded)
        decoded = self.decoder(encoded)
        #print(decoded)
        return decoded
    
    def pipeline_fit(self, train_data, epochs):
        history = self.fit(train_data, train_data, epochs=epochs, validation_split=0.2, shuffle=True)
        #reconstructions = self.predict(train_data)
        #train_loss = mae(reconstructions, train_data)
        return history

    def pipeline_predict(self, test_data, contamination):
        reconstructions = self.predict(test_data)
        test_loss = mae(test_data, reconstructions) # 1 = anomaly (same as data)
        contamination = contamination + 0.3 * (1 - contamination) #corrective factor to contamination to increase autoencoder recall
        threshold = np.percentile(test_loss, 100 - 100 * contamination)
        predictions = less(threshold, test_loss) # if threshold < loss, then we return a 1, as it's an anomaly, else return 0
        return test_loss.numpy(), predictions.numpy()
