from matplotlib import pyplot as plt
from pandas import DataFrame, Series, concat
from time import time
from autoencoder.aecExtraFeatures import getZVector
from turtleIsolationForests.printResults import calc_confusion, calc_f1, print_by_result, get_auroc_value
import numpy as np
import typing

def run_pipeline(X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, 
                 autoenc: any, iForest: any, intermediatePrint=False, printHistory=False):
    train_labels_np = train_labels.to_numpy()
    test_labels_np = test_labels.to_numpy()
    X_train_ae = X_train[train_labels_np] #autoencoder trains only on normal data
    history = autoenc.pipeline_fit(X_train_ae)
    X_train_forest_np = addZToData(X_train, autoenc)
    iForest.fit(X_train_forest_np, train_labels)
    start_time = time()
    ae_scores, ae_predictions = autoenc.pipeline_predict(X_test, test_labels_np)
    ae_time = time() - start_time
    ae_TA, ae_FA, ae_FN, ae_TN = calc_confusion(ae_predictions, test_labels)
    if printHistory:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Training Loss & Validation Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim([0,.25])
        plt.legend(["Loss", "Validation Loss"])
        plt.show()
    if intermediatePrint:
        ae_auroc = get_auroc_value(ae_scores, test_labels_np)
        ae_precision, ae_recall, ae_f1 = calc_f1(ae_TA, ae_FA, ae_FN, ae_TN)
        print("Autoencoder Results")
        print_by_result(ae_TA, ae_FA, ae_FN, ae_TN, ae_precision, ae_recall, ae_f1)
        print("auroc: " + str(ae_auroc))
        print("test set prediction time: " + str(ae_time))
        print("")
    X_test_forest_np = addZToData(X_test[ae_predictions], autoenc).to_numpy()
    test_labels_forest_np = test_labels_np[ae_predictions]
    start_time = time()
    if_scores, if_predictions = iForest.predict(X_test_forest_np, test_labels_forest_np)
    if_time = time() - start_time
    if_TA, if_FA, if_FN, if_TN = calc_confusion(if_predictions, test_labels_forest_np)
    if intermediatePrint:
        print("Isolation Forest Results")
        if_precision, if_recall, if_f1 = calc_f1(if_TA, if_FA, if_FN, if_TN)
        if_auroc = get_auroc_value(if_scores, test_labels_forest_np)
        print_by_result(if_TA, if_FA, if_FN, if_TN, if_precision, if_recall, if_f1)
        print("auroc: " + str(if_auroc))
        print("test set prediction time: " + str(if_time))
        print("")
    TA = if_TA
    FA = if_FA
    FN = ae_FN + if_FN
    TN = ae_TN + if_TN
    precision, recall, f1 = calc_f1(TA, FA, FN, TN)
    print("Pipeline Results:")
    print_by_result(TA, FA, FN, TN, precision, recall, f1)
    print("Stage 1 prediction time: " + str(ae_time))
    print("Percentage of data passed to stage 2: " + str(sum(ae_predictions) / len(ae_predictions)))
    print("Stage 2 prediction time: " + str(if_time))

def addZToData(data: DataFrame, model) -> DataFrame:

    #data_with_Z = [addZToPrediction(model, datum) for datum in data]
    z_data = data.apply(getZVector, axis=1, raw=True, result_type='expand', model=model)

    #data_with_Z_rf = []
    #for i in range(len(data_with_Z)):
    #    data_with_Z_rf.append(np.append(data[:][:][i].numpy().reshape(1,46).squeeze(), data_with_Z[i]))

    #def z_rf_function(i):
    #    return np.append(data[:][:][i].numpy().reshape(1,46).squeeze(), data_with_Z[i])
    
    #data_with_Z_rf = [np.append(data[:][:][i].numpy().reshape(1,46).squeeze(), datum_with_Z) for datum_with_Z in data_with_Z]

    #return np.ndarray(data_with_Z_rf)
    #return np.fromfunction(z_rf_function, data_with_Z.shape)
    return data.join(z_data)

#def addZToPrediction(data_point: Series, model):
def getZVector(data_point: np.ndarray, model):
    encoded = model.encoder(data_point)
    reconstruction = model.decoder(encoded)

    Z_features = getZVector(data_point, reconstruction, encoded)

    #Z_features_tensor = tf.convert_to_tensor(Z_features, dtype=tf.float32)
    #data_point = tf.convert_to_tensor(data_point, dtype=tf.float32)

    #data_point = tf.concat([data_point, Z_features_tensor], 1)

    return concat(data_point, Z_features)
