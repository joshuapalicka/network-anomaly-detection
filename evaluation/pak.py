def pak(anomaly_segment_list, ground_truth, threshold,  k):
    allAboveThreshold = True

    for item in anomaly_segment_list:
        if item <= threshold:
            allAboveThreshold = False

    if allAboveThreshold:
        return True

    numCorrectlyDetected = 0

    for i in range(len(anomaly_segment_list)):
        if anomaly_segment_list[i] == ground_truth[i]:
            numCorrectlyDetected += 1

    return numCorrectlyDetected / len(anomaly_segment_list) > k