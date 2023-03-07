from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from time import time
from autoencoder.aecExtraFeatures import getZVector
from turtleIsolationForests.printResults import calc_confusion, calc_f1, print_by_result, get_auroc_value
import numpy as np
import typing

def run_pipeline(X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, 
                 autoenc: any, iForest: any, intermediatePrint=False, printHistory=False, epochs=200):
    train_labels_np = train_labels.to_numpy()
    test_labels_np = test_labels.to_numpy()
    X_train_ae = X_train[train_labels_np] #autoencoder trains only on normal data
    history = autoenc.pipeline_fit(X_train_ae, epochs=epochs)
    X_train_forest = DataFrame(addZToData(X_train, autoenc))
    iForest.fit(X_train_forest, train_labels)
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
    X_test_forest = DataFrame(addZToData(X_test[ae_predictions], autoenc))
    test_labels_forest_np = test_labels_np[ae_predictions]
    start_time = time()
    if_scores, if_predictions = iForest.predict(X_test_forest, test_labels_forest_np)
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
    data = data.to_numpy()
    data_with_Z = []
    for i in range(len(data)):
        data_with_Z.append(addZToPoint(model, data[i:i+1]))
    data_with_Z_np = np.stack(data_with_Z)
    return data_with_Z_np

def addZToPoint(model, data_point: np.ndarray):
    encoded = model.encoder(data_point)
    reconstruction = model.decoder(encoded)
    Z_features = getZVector(data_point, reconstruction, encoded)
    return np.concatenate((data_point[0], Z_features))
