# Network Anomaly Detection

# Seattle University Data Science Capstone Project - Winter Quarter 2023


| Name  | Role |
| --------  | -------- |
| Joshua Palicka | Student |
| Alexander Sheardown | Student |
| James Talbott | Student |
| Dr. Wan Bae | Advisor |


## Introduction
The world of technology has allowed us to be more connected but it also makes us more vulnerable to cyber threats, making advanced threat detection more important than ever. We build on recent work by Carrera et al [1], further testing their two-phase anomaly detection process. This paper proposes that a model that predicts quickly can be used to reduce the incoming data load for a second, slower, more accurate model, thereby achieving faster and more accurate predictions than either model alone. To improve the model pipeline, we introduce new data augmentations and test many combinations. We use autoencoder variants for the first phase and isolation forest variants for the second. We consider a 2-phase pipeline to have outperformed its component models if the pipeline predicts faster than the phase 2 model without significant loss of score and scores higher than the phase 1 model without significant loss of prediction speed. Our results confirm that the 2-phase prediction pipeline, given the right conditions, outperforms its components under this guideline.

## Dataset
We chose the NSL_KDD dataset as it is one of the datasets used in [1], and is well-documented. It contains packet data for network analysis, and is meant as a train and test dataset for use in anomaly detection. Each data point is classified as either 0 (normal) or 1 (anomalous), which allows for binary classification on whether the data point is an anomaly or not.

## Models
We used autoencoder variants for the first phase and isolation forest variants for the second. The isolation forest variants we used are Isolation Forest, first defined in [3], which is used in [1], Extended Isolation Forest (EIF) [4], SCiForest (SCiF) [5], and FBiF [6]. The non-anomalous training data was preprocessed with a robust autoencoder [8].

## Running the Models
1. Install packages with `pip install requirements.txt`
2. Run either `Combined_Robust_Pipeline.ipynb` or `Combined_Pipeline.ipynb`, depending on if you'd like to use the robust autoencoder to preprocess the dataset or not. Results will print in the corresponding notebook file once complete.

## Conclusion
Our study investigated the pipelining of network traffic anomaly detection models to improve speed and accuracy over their components. We first preprocessed the normal data points in the NSL-KDD dataset using a robust autoencoder to reduce noise and thus improve data quality. Subsequently, we implement a 2-phase approach by combining an autoencoder with an isolation forest in a pipeline. This methodology leads to significant time savings compared to using an isolation forest alone, but still has room to further improve the overall F1 score as compared to the phase 1 autoencoder. The potential for such improvement is shown to potentially lie in underlying qualities of the autoencoder that are not yet well examined.

## References
References
[1]: Carrera, Francesco, et al. "Combining unsupervised approaches for near real-time network traffic anomaly detection." Applied Sciences 12.3 (2022): 1759.

[2]: Iglesias, Félix, and Tanja Zseby. "Analysis of network traffic features for anomaly detection." Machine Learning 101 (2015): 59-84.

[3]: Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008.

[4]: Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner. "Extended isolation forest." IEEE Transactions on Knowledge and Data Engineering 33.4 (2019): 1479-1489.

[5]: Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "On detecting clustered anomalies using SCiForest." Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2010, Barcelona, Spain, September 20-24, 2010, Proceedings, Part II 21. Springer Berlin Heidelberg, 2010.

[6]: Choudhury, Jayanta, et al. "Hypersphere for Branching Node for the Family of Isolation Forest Algorithms." 2021 IEEE International Conference on Smart Computing (SMARTCOMP). IEEE, 2021.

[7]: J. Choudhury and C. Shi, "Enhanced Performance of Finitie [sic] Boundary Isolation Forest (FBIF) for Datasets with Standard Distribution Properties," 2022 International Conference on Electrical, Computer and Energy Technologies (ICECET), Prague, Czech Republic, 2022, pp. 1-5, doi: 10.1109/ICECET55527.2022.9873022.

[8]: Zong, Bo, et al. "Deep autoencoding gaussian mixture model for unsupervised anomaly detection." International conference on learning representations. 2018.

