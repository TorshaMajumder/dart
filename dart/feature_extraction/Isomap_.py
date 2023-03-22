#
# Import all the dependencies
#
import os
import pickle
import warnings
from sklearn.manifold import Isomap

warnings.filterwarnings("ignore", category=RuntimeWarning)

#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class IsoMap(object):
    """

    Isomap is a feature extraction tool for dimensionality reduction

    Parameters
    ----------

    X_train: numpy ndarray (default = None)
        training data set

    lc_type: string
        type of light curves (transits or transients)

    n_neighbors: int(default = 10)
        Number of neighbors to consider for each point.

    n_features: int (default = 10)
        number of features in the latent space
        Note: This parameter is equivalent to isomap's n_components

    metric: String (default = 'cosine')
        The metric to use when calculating distance between instances
        in a feature array. If metric is a string or callable,
        it must be one of the options allowed by sklearn.metrics.pairwise_distances
        for its metric parameter.

    n_jobs: int (default = -1)
        number of parallel jobs to run


    """

    def __init__(self, lc_type=None, n_features=5, n_jobs=-1,
                 metadata=None, n_neighbors=10, metric="cosine"):

        self.labels = None
        self.type = lc_type
        self.metric = metric
        self.n_jobs = n_jobs
        self.estimator = None
        self.metadata = metadata
        self.n_features = n_features
        self.n_neighbors = n_neighbors

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")

            if self.type == "transits":
                raise NotImplementedError(f"\nNotImplementedError: Please specify the -- type -- as 'transients'!\n"
                                          f"'{self.type}' is not implemented yet!\n")
        except Exception as e:
            print(e)
            exit()

        try:

            if self.metadata:
                for i in self.metadata:
                    if i not in ["max_flux", "mwebv"]:
                        raise ValueError(f"\nValueError: '{i}' is an invalid metadata!"
                                         f"\nPlease provide parameters as - 'max_flux' for maximum flux, 'mwebv' "
                                         f"for Milky Way extinction.")
        except Exception as e:
            print(e)
            exit()

    def fit_transform(self, X_train=None):

        """
        Fits the data to Isomap estimator

        """

        try:

            self.estimator = Isomap(n_jobs=self.n_jobs,
                                    metric=self.metric,
                                    n_components=self.n_features,
                                    n_neighbors=self.n_neighbors)

            transformed_data = self.estimator.fit_transform(X_train, y=None)
            #
            # Create '/latent_space_data/{type}/' folder if it does not exists already
            #
            if not os.path.exists(f"../latent_space_data/{self.type}"):
                os.makedirs(f"../latent_space_data/{self.type}")
            #
            # Create dictionary to add metadata
            #
            data = {'transformed_data': transformed_data, 'labels': self.labels}
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(f"../latent_space_data/{self.type}/isomap.pickle", 'wb') as file:
                pickle.dump(data, file)
            #
            #
            #
            print(f"\nIsomap's latent space data is extracted and stored "
                  f"in -- /latent_space_data/{self.type} -- folder!\n")

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return


if __name__ == '__main__':

    i_map = IsoMap(lc_type="transients", n_features=10)
    i_map.fit_transform(X_train=None)



