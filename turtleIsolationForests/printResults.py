def print_results(predictions):
    anomalies = predictions.loc[predictions['is_normal'] == 0]
    true_anomalies = anomalies.loc[predictions['predicted_as_anomaly'] == True]
    false_anomalies = anomalies.loc[predictions['predicted_as_anomaly'] == False]

    normals = predictions.loc[predictions['is_normal'] != 0]
    true_normals = normals.loc[predictions['predicted_as_anomaly'] == True]
    false_normals = normals.loc[predictions['predicted_as_anomaly'] == False]

    TA = len(true_anomalies)
    FA = len(false_anomalies)
    FN = len(false_normals)
    TN = len(true_normals)

    precision = TA / (TA + FA)
    recall = TA / (TA + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print("true anomalies: " + str(TA))
    print("false anomalies: " + str(FA))
    print("false normals: " + str(FN))
    print("true normals: " + str(TN))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1-score: " + str(f1))
