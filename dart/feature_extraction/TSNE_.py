#
# Import all the dependencies
#
import os
import pickle
from sklearn.manifold import TSNE
#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class TSNe_(object):

    """

    TSNE is a feature extraction tool for dimensionality reduction


    Parameters
    ----------

    lc_type: string
        type of light curves (transits or transients)

    n_features: int (default = 12)
        number of features in the latent space


    """

    def __init__(self, lc_type=None, n_features=2):

        self.type = lc_type
        self.estimator = None
        self.n_features = n_features

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

    def fit_transform(self, X_train=None, visualize=True):

        """
        Fits and Transforms the data using TSNE

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
            # Initialize the tsne estimator
            #
            self.estimator = TSNE(n_components=self.n_features, learning_rate='auto',
                                  init='random', perplexity=5)
            #
            # Fit and transform the data
            #
            transformed_data = self.estimator.fit_transform(X_train)

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return

        if visualize:
            return transformed_data

        else:
            #
            #
            #
            print(f"\nData has been fitted and transformed using TSNE!\n")
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(f"../latent_space_data/{self.type}/tsne.pickle", 'wb') as file:
                pickle.dump(transformed_data, file)
            #
            #
            #
            print(f"\nTSNE latent space data is extracted and stored "
                  f"in -- /latent_space_data/{self.type} -- folder!\n")


if __name__ == '__main__':

    tsne = TSNe_(lc_type="transients", n_features=2)
    tsne.fit_transform(X_train=None, visualize=False)

