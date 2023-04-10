#
# Import all the dependencies
#
import os
import pickle
from sklearn.neighbors import LocalOutlierFactor
from TESS.datasets.transients import load_latent_space

#
# Create '/results/anomaly_detection' folder if it does not exists already
#
if not os.path.exists('../results/anomaly_detection'):
    os.makedirs('../results/anomaly_detection')


class LOF(object):


    """
    LocalOutlierFactor is an unsupervised anomaly detection technique
    used for the light curves from NASA's TESS telescope.

    The anomaly scores are available in the folder -
        -- ../results/anomaly_detection/transients
        -- ../results/anomaly_detection/transits

    Parameter
    --------
    n_neighbors: int (default = 10)
        Number of neighbors to use by default for kneighbors queries.

    metric: string (default = "cosine")
        Metric to use for distance computation.

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of outliers in the data set

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', 'umap', 'isomap',and 'k_pca'


    """

    def __init__(self, X=None, n_neighbors=10, metric="cosine",
                 contamination=0.1, labels=None, lc_type=None, extract_type=None):

        self.X = X
        self.type = lc_type
        self.metric = metric
        self.labels = labels
        self.estimators = None
        self.anomaly_index = None
        self.anomaly_score = None
        self.n_neighbors = n_neighbors
        self.extract_type = extract_type
        self.contamination = contamination

        try:
            if self.type not in ["transits", "transients"]:
                raise TypeError(f"\nTypeError: '{self.type}' is not a valid type!"
                                f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

        try:
            if self.extract_type not in ["k_pca", "tsfresh", "vae", "isomap", "umap"]:
                raise TypeError(f"\nTypeError: '{self.extract_type}' is not a valid type!"
                                f"\nPlease provide the type as - 'k_pca' , 'tsfresh', 'umap','isomap',or 'vae'")
        except Exception as e:
            print(e)
            exit()



    def fit(self, X, y=None):

        """
        Fits the data to LocalOutlierFactor

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
            not used, present here for consistency by convention.
        """

        self.estimators = LocalOutlierFactor(n_neighbors=self.n_neighbors, metric=self.metric,
                                             contamination=self.contamination)
        self.estimators.fit(X)



    def predict(self, X, y=None):

        """
        Predicts the data using LocalOutlierFactor

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
            not used, present here for consistency by convention.

        Returns
        -------
        anomaly_index: None
            not used, present here for consistency by convention.

        anomaly_score: ndarray
            anomaly score of the data samples (lower scores are more anomalous and higher scores are more normal)

        """

        self.estimators.predict(X)
        self.anomaly_score = self.estimators.negative_outlier_factor_
        #
        # Prepare a dictionary using IAU Name, anomaly index, and anomaly score
        #
        anomaly_labels = {'labels': self.labels, 'anomaly_index': None,
                          'anomaly_score': self.anomaly_score}
        #
        # Create '/results/anomaly_detection/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../results/anomaly_detection/{self.type}"):
            os.makedirs(f"../results/anomaly_detection/{self.type}")
        #
        # Store the file in -- '/results/anomaly_detection/{type}/' folder
        #
        with open(f"../results/anomaly_detection/{self.type}/lof_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(anomaly_labels, file)
        #
        #
        #
        print(f"\nAnomaly scores are generated and stored "
              f"in -- /results/anomaly_detection/{self.type} -- folder!\n")

        return None, self.anomaly_score

    def get_anomaly_score(self, X):

        """
        Get the anomaly score using LocalOutlierFactor

        Parameter
        ---------
        X: ndarray
            training data set

        Returns
        -------
        anomaly_score: ndarray
            anomaly score of the data samples (lower scores are more anomalous and higher scores are more normal)

        """
        self.estimators.fit_predict(X)
        self.anomaly_score = self.estimators.negative_outlier_factor_

        return self.anomaly_score


if __name__ == '__main__':

    data = load_latent_space(extract_type='umap')
    X_train, labels = data['data'], data['labels']
    params = { 'X': X_train,
               'contamination': 0.1,
               'n_neighbors': 10,
               'metric': 'cosine',
               'labels': labels,
               'lc_type': 'transients',
               'extract_type': 'umap'
               }

    lof = LOF(**params)
    lof.fit(X_train)
    anomaly_index, anomaly_score = lof.predict(X_train)
    #anomaly_score = lof.get_anomaly_score(X_train)





