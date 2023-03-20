#
# Import all the dependencies
#
import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from TESS.datasets.transients import load_latent_space

#
# Create '/results/anomaly_detection' folder if it does not exists already
#
if not os.path.exists('../results/anomaly_detection'):
    os.makedirs('../results/anomaly_detection')

class Isolation_Forest(object):

    """
    Isolation Forest is an anomaly detection technique
    used for the light curves from NASA's TESS telescope.

    The anomaly scores are available in the folder -
        -- ../results/anomaly_detection/transients
        -- ../results/anomaly_detection/transits

    Parameter
    --------
    n_features: int
        number of features in the data sample

    max_features: int or float (default = 1.0)
        number of features to draw from X to train each base estimator

    bootstrap: bool (default=False)
        if True, individual trees are fit on random subsets of the training
        data sampled with replacement; if False, sampling without replacement is performed

    max_samples: “auto”, int or float (default=256)
        number of samples to draw from X to train each base estimator

    n_samples: int or float
        number of data samples

    n_estimators: int (default=100)
        number of base estimators in the ensemble

    random_state: int (default=0)
        controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of outliers in the data set

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', 'isomap',and 'k_pca'


    """

    def __init__(self, X=None, n_features=None, max_features=1.0, bootstrap=False,
                 max_samples=256, n_samples=None, n_estimators=100, random_state=0,
                 contamination=0.1, labels=None, lc_type=None, extract_type=None):

        self.X = X
        self.n_features = n_features
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_samples = n_samples
        self.n_estimators = n_estimators
        self.estimators = None
        self.random_state = random_state
        self.contamination = contamination
        self.labels = labels
        self.type = lc_type
        self.extract_type = extract_type
        self.anomaly_index = None
        self.anomaly_score = None

        try:
            if self.type not in ["transits", "transients"]:
                raise TypeError(f"\nTypeError: '{self.type}' is not a valid type!"
                                f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

        try:
            if self.extract_type not in ["k_pca", "tsfresh", "vae", "isomap"]:
                raise TypeError(f"\nTypeError: '{self.extract_type}' is not a valid type!"
                                f"\nPlease provide the type as - 'k_pca' , 'tsfresh', 'isomap',or 'vae'")
        except Exception as e:
            print(e)
            exit()


    def fit(self, X, y=None):

        """
        Fits the data to Isolation Forest

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
            not used, present here for consistency by convention.
        """

        self.estimators = IsolationForest(n_estimators=self.n_estimators,
                                            max_features=self.max_features,
                                            max_samples=self.max_samples,
                                            bootstrap=self.bootstrap,
                                            random_state=self.random_state,
                                            contamination=self.contamination)


        self.estimators.fit(X)



    def predict(self, X, y=None):

        """
        Predicts the data using Isolation Forest

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
            not used, present here for consistency by convention.

        Returns
        -------
        anomaly_index: ndarray
            index of the anomaly where the cluster value is -1

        anomaly_score: ndarray
            anomaly score of the data samples (negative scores for anomalies)

        """

        clusters = self.estimators.predict(X)
        self.anomaly_index = np.where(clusters == -1)
        self.anomaly_index = self.anomaly_index[0]
        self.anomaly_score = self.estimators.decision_function(X)
        #
        # Prepare a dictionary using IAU Name, anomaly index, and anomaly score
        #
        anomaly_labels = {'labels': self.labels, 'anomaly_index': self.anomaly_index,
                          'anomaly_score': self.anomaly_score}
        #
        # Create '/results/clustering/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../results/anomaly_detection/{self.type}"):
            os.makedirs(f"../results/anomaly_detection/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/anomaly_detection/{self.type}/iforest_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(anomaly_labels, file)
        #
        #
        #
        print(f"\nAnomaly scores are generated and stored "
              f"in -- /results/anomaly_detection/{self.type} -- folder!\n")

        return self.anomaly_index, self.anomaly_score

    def get_anomaly_score(self, X):

        """
        Get the anomaly score using Isolation Forest

        Parameter
        ---------
        X: ndarray
            training data set

        Returns
        -------
        anomaly_score: ndarray
            anomaly score of the data samples (negative scores for anomalies)

        """

        self.anomaly_score = self.estimators.decision_function(X)

        return self.anomaly_score


    def get_feature_importance(self):

        """
        Get the feature importance using Isolation Forest

        Returns
        -------
        sorted_set: ndarray
            feature importance of the features

        """

        score_matrix = np.zeros((self.n_features, self.n_estimators))
        for i, estimator in enumerate(self.estimators.estimators_):
            features_score=estimator.feature_importances_
            score_matrix[:,i]=features_score

        feature_importance=np.sum(score_matrix, axis=1)
        features = np.arange(self.n_features)
        sorted_set = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)
        return sorted_set


if __name__ == '__main__':

    data = load_latent_space(extract_type='tsfresh')
    X_train, labels = data['data'], data['labels']
    params = { 'X': X_train,
               'n_features': X_train.shape[1],
               'max_samples': 1000,
               'contamination': 0.02,
               'max_features': 1.0,
               'bootstrap': False,
               'random_state': 0,
               'n_estimators': 100,
               'n_samples': X_train.shape[0],
               'labels': labels,
               'lc_type': 'transients',
               'extract_type': 'tsfresh'
               }

    IForest = Isolation_Forest(**params)
    IForest.fit(X_train)
    anomaly_index, anomaly_score = IForest.predict(X_train)
    IForest.get_anomaly_score(X_train)
    features = IForest.get_feature_importance()




