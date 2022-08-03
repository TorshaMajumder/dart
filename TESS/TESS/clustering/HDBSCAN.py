#
# Import all the dependencies
#
import os
import pickle
import hdbscan
from TESS.datasets.transients import load_latent_space

#
# Create '/results/clustering' folder if it does not exists already
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

    gen_min_span_tree: bool (default=True)
        build the minimal spanning tree

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of outliers in the data set

    labels: string
        IAU Name or TIC ID

    type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', and 'k_pca'


    """

    def __init__(self, X=None, min_cluster_size=10, min_samples=10, gen_min_span_tree=True,
                 labels=None, type=None, extract_type=None, contamination=0.1):
        self.X = X
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.gen_min_span_tree = gen_min_span_tree
        self.estimator = None
        self.clusters = None
        self.anomaly_score = None
        self.anomaly_index = None
        self.labels = labels
        self.n_samples = None
        self.type = type
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
            if self.extract_type not in ["k_pca", "tsfresh", "vae"]:
                raise TypeError(f"\nTypeError: '{self.extract_type}' is not a valid type!"
                                f"\nPlease provide the type as - 'k_pca' , 'tsfresh', or 'vae'")
        except Exception as e:
            print(e)
            exit()



    def fit(self, X, y=None):

        """
        Fits the data to Birch estimator

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
            not used, present here for consistency by convention.
        """
        self.n_samples = X_train.shape[0]
        self.estimator = hdbscan.HDBSCAN()

        self.estimator.fit(X)


    def predict(self, X, y=None):

        """
        Predicts the data using Birch estimator

        Parameter
        ---------
        X: ndarray
            training data set

        y: Ignored
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
        #
        # Select the (100-(idx*100))th percentile of the weirdness score
        #
        idx = int(self.n_samples*self.contamination)
        #
        # Store the cluster ids, anomaly index and score
        #
        self.clusters = self.estimator.labels_
        self.anomaly_score = self.estimator.outlier_scores_
        self.anomaly_index = sorted(range(len(self.anomaly_score)), key=lambda i: self.anomaly_score[i])[-idx:]
        #
        # Prepare a dictionary using IAU Name, cluster index, anomaly index and anomaly score
        #
        labels_cluster_anomaly = {'labels': self.labels, 'clusters': self.clusters,
                                  'anomaly_score': self.anomaly_score, 'anomaly_index': self.anomaly_index}
        #
        # Create '/results/clustering/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../results/clustering/{self.type}"):
            os.makedirs(f"../results/clustering/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/clustering/{self.type}/hdbscan_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(labels_cluster_anomaly, file)
        #
        #
        #
        print(f"\nClusters are generated and stored "
              f"in -- /results/clustering/{self.type} -- folder!\n")

        return self.clusters, self.anomaly_score, self.anomaly_index


if __name__ == '__main__':

    data = load_latent_space(extract_type='tsfresh')
    X_train, labels = data['data'], data['labels']
    hdbscan_ = HDBSCAN_(labels=labels, type='transients', extract_type='tsfresh', contamination=0.1)
    hdbscan_.fit(X_train)
    clusters, anomaly_score, anomaly_index = hdbscan_.predict(X_train)

