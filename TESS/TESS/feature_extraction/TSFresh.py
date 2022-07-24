#
# Import all the dependencies
#
import os
import pickle
import numpy as np
import pandas as pd
import tsfresh as tf
from TESS.feature_extraction.KernelPCA import Kernel_PCA
from TESS.anomaly_detection.UnsupervisedRF import UnsupervisedRandomForest
#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')

class TSFresh(object):

    """

    TSFresh is a feature extraction tool applied with an Unsupervised Random Forest Classifier
    for dimensionality reduction of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle

    Parameters
    ----------

    type: string
        type of light curves (transits or transients)

    n_features: int (default = 10)
        number of features in the latent space

    tsfresh_data: pandas-dataframe
        tsfresh compatible dataframe

    """



    def __init__(self, type=None, n_features=10):

        self.type = type
        self.n_features = n_features
        self.tsfresh_data = None


    def generate_data(self, path=None):

        """
        Generates TSFresh package compatible dataframe

        Parameters
        ----------
        path: string
            the file location of the light curves

        """
        #
        #
        #
        data = list()
        #
        # Load the pickle file
        #
        with open(path, 'rb') as file:
            lightcurves = pickle.load(file)
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
                columns = ['ID', 'IAU Name', 'TIME', 'FLUX']
                for idx in range(len(flux)):
                    for i, lc in enumerate(zip(time, flux[idx])):
                        data.append([idx, meta_data[idx], time[i], flux[idx][i]])

            elif self.type == "transits":
                phase = lightcurves['phase']
                columns = ['ID', 'TOI', 'PHASE', 'FLUX']
                for idx in range(len(flux)):
                    for i, lc in enumerate(zip(phase, flux[idx])):
                        data.append([idx, meta_data[idx][1], phase[i], flux[idx][i]])
            else:
                raise ValueError(f"\nValueError: Please specify the -- type -- of the file!\n"
                                 f"It should be transients/transits type. '{self.type}' is an invalid type!\n")

        except Exception as e:
            print(e)
            return

        #
        # Store the data in a pandas-dataframe
        # with the column names specified in the variable -- columns
        #
        tsfresh_df = pd.DataFrame(data, columns=columns)
        #
        # Store the dataframe
        #
        self.tsfresh_data = tsfresh_df

        print(f"\nTSFresh data is created!\n")



    def extract_features(self, path=None):

        """
        Generates TSFresh extracted features

        Parameters
        ----------
        path: string
            the file location where the features are stored

        """
        #
        #
        #
        try:
            #
            # Check the type of the file
            #
            if self.type == "transients":
                #
                # column_id will be 'IAU Name' for transients type
                #
                extracted_features = tf.extract_features(self.tsfresh_data, column_id='IAU Name', column_sort='TIME')

            elif self.type == "transits":
                #
                # column_id will be 'TOI' for transits type
                #
                extracted_features = tf.extract_features(self.tsfresh_data, column_id='TOI', column_sort='PHASE')

            else:
                raise ValueError(f"\nValueError: Please check the -- type -- of the file!\n"
                                 f"It should be transients/transits type. '{self.type}' is an invalid type!\n")

            #
            # Drop columns and rows that have NaNs
            # Drop any [np.inf, -np.inf] with NaNs and
            # Fill NaNs with a high positive integer
            #
            extracted_features = extracted_features.dropna(axis='columns')
            extracted_features = extracted_features.dropna()
            extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            extracted_features.fillna(99999, inplace=True)
            #
            # Store the files in {path}
            #
            with open(path, 'wb') as file:
                pickle.dump(extracted_features, file)

        except Exception as e:
            print(e)

        print(f"\nTSFresh features are extracted and stored in -- {path}!\n")


    def get_important_features(self, path=None, method="URF"):

        """
        Generates Unsupervised Random Forest extracted important features

        Parameters
        ----------
        path: string
            the file location of the TSFresh extracted features

        method: string (default = URF)
            method to extract 'n_features' using - Kernel_PCA (KPCA) or Unsupervised_RF (URF)

        """
        #
        # Create '/latent_space_data/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../latent_space_data/{self.type}"):
            os.makedirs(f"../latent_space_data/{self.type}")
        #
        # Validate the method type
        #
        try:
            if method not in ["URF", "KPCA"]:
                raise ValueError(f"\nValueError: Unknown method type!\nPlease provide method as URF or KPCA.\n")
        except Exception as e:
            print(e)
            return
        #
        # Load TSFresh data
        #
        try:
            with open(path, 'rb') as file:
                tsfresh_data = pickle.load(file)
        except Exception as e:
            print(f"\nFileNotFound: Unable to load the .pickle file!\n")
            return
        #
        # Extract - n_features - using KPCA or URF
        #
        try:
            #
            # Convert the dataframe to numpy array
            #
            extracted_features = tsfresh_data.to_numpy()
            #
            #
            #
            if method == "URF":
                feature_list = list()
                #
                # Default parameters for Unsupervised RF
                #
                params = { 'X': extracted_features,
                   'n_features': extracted_features.shape[1],
                   'max_depth': 100,
                   'min_samples_split': 3,
                   'max_features': 'log2',
                   'bootstrap': False,
                   'n_samples': extracted_features.shape[0],
                   'n_estimators': 100,
                   'random_state': 0
                   }
                #
                # Initialize Unsupervised RF classifier
                #
                URF = UnsupervisedRandomForest(**params)
                #
                # Generate synthetic data with labels
                #
                X, y = URF.generate_data()
                #
                # Fit the data to the classifier
                #
                URF.fit(X, y)
                #
                # Get the important features
                #
                features = URF.get_feature_importance(plot=False)

                for i in range(self.n_features):
                    idx, val = features[i][0], features[i][1]
                    feature_list.append(idx)
                #
                # Reshape the dataframe
                #
                extracted_df = tsfresh_data.iloc[:, feature_list]
                extracted_df = extracted_df.reset_index(drop=True)
                #
                # Store the file in -- '/latent_space_data/{type}/' folder
                #
                with open(f"../latent_space_data/{self.type}/tsfresh.pickle", 'wb') as file:
                    pickle.dump(extracted_df, file)

                print(f"\nTSFresh latent space data is extracted and stored "
                      f"in -- /latent_space_data/{self.type} -- folder!\n")

            elif method == "KPCA":
                #
                # Initialize Kernel_PCA classifier
                #
                k_pca = Kernel_PCA(X_train=extracted_features, type=self.type, n_features= self.n_features)
                #
                # Fit the data to the classifier
                #
                k_pca.fit()
                #
                # Transform the data using the classifier
                #
                k_pca.transform(tsfresh=True)
                #
                #
                #
                print(f"\nTSFresh latent space data is extracted and stored "
                      f"in -- /latent_space_data/{self.type} -- folder!\n")


        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return


if __name__ == '__main__':


    tsfresh = TSFresh(type="transients")
    tsfresh.generate_data(path="../transients/data/transients.pickle")
    tsfresh.extract_features(path="../transients/data/tsfresh_data.pickle")
    tsfresh.get_important_features(path="../transients/data/tsfresh_data.pickle", method="KPCA")






