import os
import umap
import pickle

#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class UMap(object):

    """

    UMap is a feature extraction tool for dimensionality reduction
    of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle

    Parameters
    ----------

    X_train: numpy ndarray (default = None)
        training data set

    type: string
        type of light curves (transits or transients)

    n_features: int (default = 10)
        number of features in the latent space


    """

    def __init__(self, X_train=None, type=None, n_features=2):

        self.X = X_train
        self.type = type
        self.n_features = n_features
        self.umap_estimator = None


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


    def fit_transform(self):

        """
        Fits and Transforms the data using UMAP

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
            #
            # Initialize the umap estimator
            #

            self.umap_estimator = umap.UMAP(n_components=self.n_features)
            #
            # Fit and transform the data
            #
            transformed_data = self.umap_estimator.fit_transform(self.X)

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return
        #
        #
        #
        print(f"\nData has been fitted and transformed using UMAP!\n")
        #
        # Store the file in -- '/latent_space_data/{type}/' folder
        #
        with open(f"../latent_space_data/{self.type}/umap.pickle", 'wb') as file:
            pickle.dump(transformed_data, file)
        #
        #
        #
        print(f"\nUMAP latent space data is extracted and stored "
              f"in -- /latent_space_data/{self.type} -- folder!\n")


if __name__ == '__main__':

    umap_ = UMap(type="transients", n_features=2)
    umap_.read_data(path="../transients/data/transients.pickle")
    umap_.fit_transform()





