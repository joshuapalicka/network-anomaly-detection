from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from time import time
from autoencoder.aecExtraFeatures import get_Zcs, get_Zed
from turtleIsolationForests.printResults import calc_confusion, calc_f1, print_by_result, get_auroc_value
from turtleIsolationForests.preprocessFeatures import minmax_preprocess_z_features
import numpy as np
import typing

def run_pipeline(X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, contamination: float,
                 autoenc: any, iForest: any, X_train_ae: np.ndarray, intermediatePrint=False, epochs=600):
    train_labels_np = train_labels.to_numpy()
    test_labels_np = test_labels.to_numpy()
    history = autoenc.pipeline_fit(X_train_ae, epochs=epochs)
    start_time = time()
    ae_scores, ae_predictions = autoenc.pipeline_predict(X_test, contamination)
    ae_time = time() - start_time
    ae_TA, ae_FA, ae_FN, ae_TN = calc_confusion(ae_predictions, test_labels_np)
    if intermediatePrint:
        ae_auroc = get_auroc_value(ae_scores, test_labels_np)
        ae_precision, ae_recall, ae_f1 = calc_f1(ae_TA, ae_FA, ae_FN, ae_TN)
        print("Autoencoder Results")
        print_by_result(ae_TA, ae_FA, ae_FN, ae_TN, ae_precision, ae_recall, ae_f1)
        print("auroc: " + str(ae_auroc))
        print("test set prediction time: " + str(ae_time))
        print("")
    pre_Z_cols = X_train.shape[1]
    X_train_forest = addZToData(X_train, autoenc)
    X_train_forest.iloc[:,pre_Z_cols:] = minmax_preprocess_z_features(X_train_forest.iloc[:,pre_Z_cols:])
    X_test_forest = addZToData(X_test[ae_predictions], autoenc)
    X_test_forest.iloc[:,pre_Z_cols:] = minmax_preprocess_z_features(X_test_forest.iloc[:,pre_Z_cols:])
    iForest.fit(X_train_forest, train_labels)
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
    encoded = model.encoder(data.to_numpy())
    decoded = model.decoder(encoded).numpy()
    encoded_np = encoded.numpy()

    data_col_count = data.shape[1]
    encoded_col_count = data.shape[1]

    z_col_names = ["z"+str(i) for i in range(encoded_np.shape[1])]
    data = data.join(DataFrame(encoded_np, index=data.index, columns=z_col_names))
    decoded_df = DataFrame(decoded, index=data.index)

    def makeZedCol(row):
        orig_sample = row[:data_col_count].to_numpy()
        reconstr_sample = decoded_df.loc[row.name].to_numpy()
        dot= get_Zed(orig_sample, reconstr_sample)
        return dot
    
    def makeZcsCol(row):
        orig_sample = row[:data_col_count].to_numpy()
        reconstr_sample = decoded_df.loc[row.name].to_numpy()
        return get_Zcs(orig_sample, reconstr_sample)

    data['zed'] = data.apply(makeZedCol, axis=1)
    data['zcs'] = data.apply(makeZcsCol, axis=1)

    return data
