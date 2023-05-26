#
# Import all the dependencies
#
import os
import pickle
from sklearn.svm import OneClassSVM
from dart.datasets.transients import load_latent_space


class OneClassSVM_(object):
    """
    OneClassSVM is an unsupervised anomaly detection technique
    used for the light curves from NASA's TESS telescope.

    The anomaly scores are available in the folder -
        -- ../results/anomaly_detection/transients
        -- ../results/anomaly_detection/transits

    Parameter
    --------

    kernel: string (default = "rbf")
        kernel used in OneClassSVM

    gamma: string (default = "auto")
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’

    nu: float (default=0.5)
        An upper bound on the fraction of training errors and
        a lower bound of the fraction of support vectors.
        Should be in the interval (0, 1]

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', 'umap', 'isomap',and 'k_pca'


    """

    def __init__(self, kernel='rbf', gamma='auto', nu=0.5):

        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.estimators = None
        self.anomaly_score = None

    def fit_predict(self, X_train=None, y=None):

        """
        Fits the data to OneClassSVM

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

        self.estimators = OneClassSVM(gamma=self.gamma, nu=self.nu)
        self.estimators.fit_predict(X_train)
        self.anomaly_score = self.estimators.score_samples(X_train)
        #
        # Prepare a dictionary using IAU Name, anomaly index, and anomaly score
        #
        # anomaly_labels = {'labels': self.labels, 'anomaly_index': None,
        #                   'anomaly_score': self.anomaly_score}
        # #
        # # Create '/results/anomaly_detection/{type}/' folder if it does not exists already
        # #
        # if not os.path.exists(f"../results/anomaly_detection/{self.type}"):
        #     os.makedirs(f"../results/anomaly_detection/{self.type}")
        # #
        # # Store the file in -- '/results/anomaly_detection/{type}/' folder
        # #
        # with open(f"../results/anomaly_detection/{self.type}/svm_{self.extract_type}.pickle", 'wb') as file:
        #     pickle.dump(anomaly_labels, file)
        #
        #
        #
        # print(f"\nAnomaly scores are generated and stored "
        #       f"in -- /results/anomaly_detection/{self.type} -- folder!\n")

        return self.anomaly_score


if __name__ == '__main__':

    data = load_latent_space(extract_type='umap')
    X_train = data['data']
    svm = OneClassSVM_()
    anomaly_score = svm.fit_predict(X_train)






