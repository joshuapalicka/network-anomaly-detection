from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.linalg as nplin
import shrink
import base_autoencoder as ba
from tensorflow.python.ops.numpy_ops import np_config

from shrink import l1shrink

class Deep_Autoencoder(object):
    def __init__(self, count):
        
        autoencoder = ba.AnomalyDetector(count)
        autoencoder.compile(optimizer='adam', loss='mae')
        self.autoencoder = autoencoder
        self.lambda_= 1.0
        self.error = 1.0e-7

    def fit(self, X, iteration):
       

        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)

        mu = (X.size) / (4.0 * nplin.norm(X,1))
        print ("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S
        XFnorm = nplin.norm(X,'fro')

        for it in range(iteration):

            self.L = X - self.S
            
            self.autoencoder.fit(self.L, self.L, 
                epochs=20000, batch_size=512,
                #validation_data=(self.L, self.L),
                shuffle=True)
            """
            self.autoencoder.fit(X = self.L, 
                epochs=50,
                shuffle=True)
            """
            
            encoded  = self.autoencoder.encoder(self.L).numpy()
            self.L  = self.autoencoder.decoder(encoded).numpy()
        
            print("L shape" + str(self.L.shape))
            print("X shape" + str(X.shape))
            #Might need to fix this
            
            #shrink.l1shrink.fin()
            self.S = shrink.l1shrink.shrink(self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            ## break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            ## break criterion 2: there is no changes for L and S 
            c2 = np.min([mu,np.sqrt(mu)]) * nplin.norm(LS0 - self.L - self.S) / XFnorm

            if c1 < self.error and c2 < self.error :
                print ("early break")
                break
            ## save L + S for c2 check in the next iteration
            LS0 = self.L + self.S


            DF = pd.DataFrame( self.L)
            DF['class_0_1'] = train_labels
            DF.to_csv("Robust_Autoencoder_Cleaned_Training.csv")
            DF = pd.DataFrame(self.S)
            DF['class_0_1'] = train_labels
            DF.to_csv("Noise_S.csv")


            
        return self.L , self.S

#decoded_data = autoencoder.decoder(encoded_data).numpy()


if __name__ == "__main__":
    train_dataframe = pd.read_csv('eda_simple_classification/network_data_mod_train.csv')
    #test_data = pd.read_csv('eda_simple_classification/network_data_mod_test.csv')

    np_config.enable_numpy_behavior()

    """
    train_labels = train_dataframe.xs('class', axis='columns')
    for column in train_dataframe.columns:
        if (column[0:5] == 'class'):
            train_dataframe.drop(column, axis='columns', inplace=True)

  

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_dataframe), columns = train_dataframe.columns)
    X_train = X_train.values
    """


    #frames = [train_data, test_data]

    #dataframe  = pd.concat(frames)
    #train_dataframe = train_data.values
    #dataframe.head()
    #train_dataframe

    # The last element contains the labels
    #labels = train_dataframe[:, -1]
    #print(labels)
    #print(train_dataframe)
    

    #print(test_data.columns )


    raw_data = train_dataframe.values
    train_labels = raw_data[:, -1]

    #print(train_labels)
    
    train_data = raw_data[:, 0:-1]
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)
    X_train = (train_data - min_val) / (max_val - min_val)
   

    
    #data = train_dataframe[:, 0:-1]
    #print(raw_data)

    RDA = Deep_Autoencoder(47)
    L, S = RDA.fit(X_train, 50)

    """
    RDA = Deep_Autoencoder(47)

    L, S = RDA.fit(X_train, 50)

    #Normal is 1 and 0 is abnormal
    DF = pd.DataFrame(L)
    DF['class_0_1'] = train_labels
    DF.to_csv("Robust_Autoencoder_Cleaned_Training data.csv")
    DF = pd.DataFrame(S)
    DF['class_0_1'] = train_labels
    DF.to_csv("Noise_S.csv")
    print(DF)
    """    

