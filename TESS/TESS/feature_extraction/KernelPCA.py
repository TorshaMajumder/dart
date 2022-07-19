#
# Import all the dependencies
#
import os
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error

#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')



class Kernel_PCA(object):


    """

    Kernel_PCA is a feature extraction tool for dimensionality reduction
    of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle

    Parameters
    ----------

    type: string
        type of light curves (transits or transients)

    n_features: int (default = 10)
        number of features in the latent space

    kernel: string (default = "rbf")
        kernel used for PCA

    gamma: float (default = 0.001)
        kernel coefficient for "rbf"

    alpha: float (default = 0.001)
        hyper-parameter of the ridge regression that learns
        the inverse_transform (when fit_inverse_transform=True)

    fit_inverse_transform: bool (default = True)
        learn the inverse transform for non-precomputed kernels

    n_jobs: int (default = -1)
        number of parallel jobs to run


    """

    def __init__(self, type=None, n_features=10, kernel='rbf', gamma=0.001,
                 alpha=0.001, fit_inverse_transform=True, n_jobs=-1):

        self.X = None
        self.type = type
        self.n_features = n_features
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.eigenvectors = None
        self.eigenvalues = None
        self.PCA_Estimator = None
        self.PCA_decoder = None
        self.fit_inverse_transform = fit_inverse_transform
        self.n_jobs = n_jobs



    def read_data(self, path=None):

        """
        Loads data for Kernel PCA

        Parameters
        ----------
        path: string
            the file location of the light curves

        """
        #
        # Load the pickle file
        #
        try:
            with open(path, 'rb') as file:
                lightcurves = pickle.load(file)
        except Exception as e:
            print(f"\nFileNotFound: Unable to load the .pickle file!\n")
        #
        #
        #
        try:
            flux = lightcurves['flux']
            flux_err = lightcurves['flux_err']
            meta_data = lightcurves['metadata']

            #
            # Check the type of the file
            #
            if self.type == "transients":
                time = lightcurves['time']
                self.X = flux

            elif self.type == "transits":
                phase = lightcurves['phase']
                self.X = flux

            else:
                raise ValueError(f"\nValueError: Please specify the -- type -- of the file!\n"
                                 f"It should be transients/transits type. '{self.type}' is an invalid type!\n")

        except Exception as e:
            print(e)
            return

        print(f"\nData is generated!\n")


    def fit(self):

        """
        Fits the data to Kernel PCA estimator

        """

        try:

            self.PCA_Estimator = KernelPCA(n_components=self.n_features, kernel=self.kernel,
                                            gamma=self.gamma, alpha= self.alpha,
                                            fit_inverse_transform=self.fit_inverse_transform,
                                            n_jobs=self.n_jobs)

            self.PCA_Estimator.fit(self.X, y=None)
            #
            # Store the eigenvectors and eigenvalues
            #
            self.eigenvectors = self.PCA_Estimator.eigenvectors_
            self.eigenvalues = self.PCA_Estimator.eigenvalues_

        except Exception as e:
            print(f"\nUnknownError: {e}\n")

        print(f"\nData has been fitted to K_PCA estimator!\n")


    def transform(self):

        """
        Transforms the data by Kernel PCA estimator

        """

        #
        # Create '/latent_space_data/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../latent_space_data/{self.type}"):
            os.makedirs(f"../latent_space_data/{self.type}")
        #
        #
        #
        try:
            transformed_data = self.PCA_Estimator.transform(self.X)
            self.PCA_decoder = self.PCA_Estimator.inverse_transform(transformed_data)

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
        #
        #
        #
        print(f"\nData has been transformed by K_PCA estimator!\n")
        #
        # Store the file in -- '/latent_space_data/{type}/' folder
        #
        with open(f"../latent_space_data/{self.type}/k_pca.pickle", 'wb') as file:
            pickle.dump(transformed_data, file)

        #
        #
        #
        print(f"\nKernel_PCA latent space data is extracted and stored "
              f"in -- /latent_space_data/{self.type} -- folder!\n")

    def reconstruction_loss(self):

        #
        # Calculate the reconstruction loss (MSE)
        #
        mse = mean_squared_error(self.X, self.PCA_decoder, squared=False)
        print(f"\nKernel PCA reconstruction loss: {mse}\n")



if __name__ == '__main__':

    k_pca = Kernel_PCA(type="transients")
    k_pca.read_data(path="../transients/data/transients.pickle")
    k_pca.fit()
    k_pca.transform()
    k_pca.reconstruction_loss()

