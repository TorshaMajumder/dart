#
# Import all the dependencies
#
import os
import pickle
from sklearn.cluster import Birch
from TESS.datasets.transients import load_latent_space
#
# Create '/results/clustering' folder if it does not exists already
#
if not os.path.exists('../results/clustering'):
    os.makedirs('../results/clustering')


class Birch_(object):

    """
    Birch_ is a clustering technique used for the light curves from NASA's TESS telescope.

    The cluster ids and anomaly scores are available in the folder -
        -- ../results/clustering/transients
        -- ../results/clustering/transits

    Parameter
    --------
    threshold: float (default=0.5)
        radius of the subcluster obtained by merging a new sample
        and the closest subcluster should be lesser than the threshold

    branching_factor: int (default=50)
        maximum number of CF subclusters in each node

    n_clusters: int (default=3)
        cluster size

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of outliers in the data set

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', 'isomap',and 'k_pca'


    """

    def __init__(self, X=None, threshold=0.5, branching_factor=50, n_clusters=3,
                 labels=None, lc_type=None, extract_type=None):

        self.X = X
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.estimator = None
        self.clusters = None
        self.labels = labels
        self.type = lc_type
        self.extract_type = extract_type

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
                                 f"\nPlease provide the type as - 'k_pca' , 'tsfresh', 'isomap', or 'vae'")
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

        self.estimator = Birch(n_clusters=self.n_clusters, threshold=self.threshold,
                                branching_factor=self.branching_factor)
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

        """
        #
        # Get the clusters
        #
        self.clusters = self.estimator.predict(X)
        #
        # Prepare a dictionary using IAU Name and the cluster index
        #
        cluster_labels = {'labels': self.labels, 'clusters': self.clusters}
        #
        # Create '/results/clustering/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../results/clustering/{self.type}"):
            os.makedirs(f"../results/clustering/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/clustering/{self.type}/birch_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(cluster_labels, file)
        #
        #
        #
        print(f"\nClusters are generated and stored "
              f"in -- /results/clustering/{self.type} -- folder!\n")

        return self.clusters


if __name__ == '__main__':

    data = load_latent_space(extract_type='tsfresh')
    X_train, labels = data['data'], data['labels']
    brc = Birch_(n_clusters=20, threshold=0.005, labels=labels, lc_type='transients', extract_type='tsfresh')
    brc.fit(X_train)
    clusters = brc.predict(X_train)






