#
# Import all the dependencies
#
import os
import pickle
import warnings
from sklearn.decomposition import KernelPCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class Kernel_PCA(object):
    """

    KernelPCA is a feature extraction tool for dimensionality reduction

    Parameters
    ----------

    X_train: numpy ndarray (default = None)
        training data set

    lc_type: string
        type of light curves (transits or transients)

    n_features: int (default = 10)
        number of features in the latent space
        Note: This parameter is equivalent to Kernel PCA's n_components

    kernel: string (default = "cosine")
        kernel used for PCA


    alpha: float (default = 0.001)
        hyper-parameter of the ridge regression that learns
        the inverse_transform (when fit_inverse_transform=True)

    fit_inverse_transform: bool (default = True)
        learn the inverse transform for non-precomputed kernels

    n_jobs: int (default = -1)
        number of parallel jobs to run


    """

    def __init__(self, lc_type=None, n_features=10, kernel="cosine",
                 fit_inverse_transform=True, n_jobs=-1, metadata=None):

        self.type = lc_type
        self.n_features = n_features
        self.kernel = kernel
        self.estimator = None
        self.decoded_data = None
        self.fit_inverse_transform = fit_inverse_transform
        self.n_jobs = n_jobs
        self.labels = None
        self.metadata = metadata

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
        Fits the data to Kernel PCA estimator

        """

        try:

            self.estimator = KernelPCA(n_components=self.n_features, kernel=self.kernel,
                                       fit_inverse_transform=self.fit_inverse_transform,
                                       n_jobs=self.n_jobs)

            transformed_data = self.estimator.fit_transform(X_train, y=None)
            decoder = self.estimator.inverse_transform(transformed_data)
            #
            # Create '/latent_space_data/{type}/' folder if it does not exists already
            #
            if not os.path.exists(f"../latent_space_data/{self.type}"):
                os.makedirs(f"../latent_space_data/{self.type}")
            #
            # Create dictionary to add metadata
            #
            data = {'data': transformed_data, 'labels': self.labels, 'decoded_data': decoder}
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(f"../latent_space_data/{self.type}/k_pca.pickle", 'wb') as file:
                pickle.dump(data, file)
            #
            #
            #
            print(f"\nKernel_PCA latent space data is extracted and stored "
                  f"in -- /latent_space_data/{self.type} -- folder!\n")

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return


if __name__ == '__main__':

    k_pca = Kernel_PCA(lc_type="transients", n_features=5)
    k_pca.fit_transform(X_train=None)


