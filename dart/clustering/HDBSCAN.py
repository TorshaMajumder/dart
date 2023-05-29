#
# Import all the dependencies
#
import os
import pickle
import hdbscan
from dart.datasets.transients import load_latent_space
#
# Create '/results/clustering' folder if it does not exist already
#
if not os.path.exists('../results/clustering'):
    os.makedirs('../results/clustering')


class HDBSCAN_(object):

    """
    HDBSCAN_ is a clustering technique used for the light curves from NASA's TESS telescope.

    The cluster ids and anomaly scores are available in the folder -
        -- ../results/clustering/transients
        -- ../results/clustering/transits

    Parameter
    --------
    min_cluster_size: int (default=10)
        the smallest size grouping that you wish to consider a cluster

    min_samples: int (default=10)
        the larger the value of min_samples, the more conservative the
        clustering – more points will be declared as noise, and clusters
         will be restricted to progressively more dense areas

    cluster_selection_epsilon: float (default=0.000000001)
        this parameter helps to merge clusters when there are a large
        number of micro-clusters.

    cluster_selection_method: string (default='eom' ---> Excess of Mass)
        This parameter determines how it selects flat clusters from
        the cluster tree hierarchy.
        Notes: Excess of Mass has a tendency to pick one or two large clusters
        and then a number of small extra clusters. In this situation we may
        want to re-cluster just the data in the single large cluster. Instead,
        a better option is to select 'leaf' as a cluster selection method.
        This will select leaf nodes from the tree, producing many small
        homogeneous clusters.

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of
        outliers in the data set

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', 'isomap',and 'k_pca'


    """

    def __init__(self, X=None, min_cluster_size=30, min_samples=10,
                 cluster_selection_epsilon=0.000000001, contamination=0.1,
                 labels=None, lc_type=None, extract_type=None, cluster_selection_method='eom'):

        self.X = X
        self.type = lc_type
        self.clusters = None
        self.labels = labels
        self.estimator = None
        self.n_samples = None
        self.anomaly_score = None
        self.anomaly_index = None
        self.min_samples = min_samples
        self.extract_type = extract_type
        self.contamination = contamination
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon

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
                                f"\nPlease provide the type as - 'k_pca' , 'tsfresh', 'isomap', 'umap', or 'vae'")
        except Exception as e:
            print(e)
            exit()

    def fit_predict(self, X_train=None, y_train=None):

        """
        Fits the data to HDBSCAN estimator

        Parameter
        ---------
        X_train: ndarray
            training data set

        y_train: Ignored
            not used, present here for consistency by convention.

        Returns
        -------
        clusters: ndarray
            index of the cluster each sample belongs to.

        anomaly_index: ndarray
            index of the anomalies

        anomaly_score: ndarray
            anomaly score of the data samples (the higher the score,
            the more likely the point is to be an outlier)
        """
        self.n_samples = X_train.shape[0]
        self.estimator = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                         cluster_selection_epsilon=self.cluster_selection_epsilon,
                                         cluster_selection_method=self.cluster_selection_method)

        self.estimator.fit(X_train)
        #
        # Select the (100-(idx*100))th percentile of the weirdness score
        #
        idx = int(self.n_samples * self.contamination)
        #
        # Store the cluster ids, anomaly index and score
        #
        self.clusters = self.estimator.labels_
        self.anomaly_score = self.estimator.outlier_scores_
        #
        # Store the anomaly_index with the sorted anomaly_score
        #
        self.anomaly_index = sorted(range(len(self.anomaly_score)), key=lambda i: self.anomaly_score[i])[-idx:]
        #
        # Prepare a dictionary using IAU Name, cluster index, anomaly index and anomaly score
        #
        cluster_labels = {'labels': self.labels, 'clusters': self.clusters}
        anomaly_labels = {'labels': self.labels, 'anomaly_index': self.anomaly_index,
                          'anomaly_score': self.anomaly_score}
        #
        # Create '/results/clustering/{type}/' folder if it does not exist already
        #
        if not os.path.exists(f"../results/clustering/{self.type}"):
            os.makedirs(f"../results/clustering/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/clustering/{self.type}/hdbscan_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(cluster_labels, file)
        #
        #
        #
        print(f"\nClusters are generated and stored "
              f"in -- /results/clustering/{self.type} -- folder!\n")
        #
        # Create '/results/clustering/{type}/' folder if it does not exist already
        #
        if not os.path.exists(f"../results/anomaly_detection/{self.type}"):
            os.makedirs(f"../results/anomaly_detection/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/anomaly_detection/{self.type}/hdbscan_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(anomaly_labels, file)
        #
        #
        #
        print(f"\nAnomaly scores are generated and stored "
              f"in -- /results/anomaly_detection/{self.type} -- folder!\n")

        return self.clusters, self.anomaly_score, self.anomaly_index


if __name__ == '__main__':

    data = load_latent_space(extract_type='umap')
    X_train, labels = data['data'], data['labels']
    hdbscan_ = HDBSCAN_(labels=labels, lc_type='transients', extract_type='umap')
    clusters, anomaly_score, anomaly_index = hdbscan_.fit_predict(X_train=X_train)




