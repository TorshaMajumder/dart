#
# Import all the dependencies
#
import os
import umap.umap_ as umap
import pickle
#
# Create '/latent_space_data' folder if it does not exist already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class UMap(object):

    """

    UMap is a feature extraction tool for dimensionality reduction
    of the light curves from NASA's TESS telescope.

    Parameters
    ----------
    X_train: numpy ndarray (default = None)
        training data set

    lc_type: string
        type of light curves (transits or transients)

    n_features: int (default = 2)
        number of features in the latent space


    """

    def __init__(self, lc_type=None, n_features=5, n_neighbors=50, min_dist=0.0,
                 metric="cosine", densmap=True, dens_lambda=10, spread=3):

        self.type = lc_type
        self.metric = metric
        self.spread = spread
        self.estimator = None
        self.densmap = densmap
        self.min_dist = min_dist
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.dens_lambda = dens_lambda

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

    def fit_transform(self, X_train=None, band=None, visualize=False):

        """
        Fits and Transforms the data using UMAP

        """
        #
        # Create '/latent_space_data/{type}/' folder if it does not exist already
        #
        if not os.path.exists(f"../latent_space_data/{self.type}"):
            os.makedirs(f"../latent_space_data/{self.type}")
        #
        #
        #
        try:
            if visualize:
                #
                # Initialize the umap estimator
                #
                self.estimator = umap.UMAP(n_components=self.n_features, random_state=42)
                #
                # Fit and transform the data
                #
                transformed_data = self.estimator.fit_transform(X_train)
                return transformed_data

            else:
                params_umap = {
                    "metric": self.metric,
                    "spread": self.spread,
                    "densmap": self.densmap,
                    "min_dist": self.min_dist,
                    "n_neighbors": self.n_neighbors,
                    "dens_lambda": self.dens_lambda,
                    "n_components": self.n_features
                    }
                self.estimator = umap.UMAP(**params_umap)
                transformed_data = self.estimator.fit_transform(X_train)
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

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return


if __name__ == '__main__':

    path = f"../transients/flux.pickle"
    with open(path, 'rb') as f:
        data = pickle.load(f)
    umap_ = UMap(lc_type="transients")
    umap_.fit_transform(X_train=data["flux"], visualize=False)






