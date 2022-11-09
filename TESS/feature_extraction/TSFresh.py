#
# Import all the dependencies
#
import os
import re
import pickle
import numpy as np
import pandas as pd
import tsfresh as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from TESS.anomaly_detection.UnsupervisedRF import UnsupervisedRandomForest
#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class TSFresh(object):

    """

    TSFresh is a feature extraction tool applied with an Unsupervised Random Forest Classifier (URF)
    or Linear PCA (PCA) for dimensionality reduction of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle

    Parameters
    ----------

    lc_type: string
        type of light curves (transits or transients)

    n_features: int (default = 10)
        number of features in the latent space


    """

    def __init__(self, lc_type=None, n_features=10, passbands=["tess"], path=None, metadata=None):

        self.type = lc_type
        self.path = path
        self.passbands = passbands
        self.metadata = metadata
        self.n_features = n_features
        self.tsfresh_data = None
        self.labels = None
        self.mwebv = None
        self.max_flux = None

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\nValueError: '{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'.")

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

        try:

            for i in self.passbands:
                if i not in ["tess", "r", "g"]:
                    raise ValueError(f"\nValueError: '{i}' is an invalid passband!"
                                     f"\nPlease provide passbands as - 'tess', 'r' for ZTF r-band, 'g' for ZTF g-band.")
        except Exception as e:
            print(e)
            exit()

    def generate_data(self):

        """
        Generates TSFresh package compatible dataframe

        Parameters
        ----------
        path: string
            the location of the light curves

        Returns
        -------
        tsfresh_data: pandas-dataframe
            TSFresh compatible dataframe with the columns in sequence -

        """
        mwebv_list, max_flux_list = list(), list()

        #
        # Store the list of files from the path
        #
        scaler = MinMaxScaler()
        filename = os.listdir(self.path)
        #
        # Declare all the variables
        #
        curve_range = (-30, 70)
        maskval, interval_val, n_bands = 0.0, 0.5, len(self.passbands)
        max_flux = np.zeros(shape=(len(filename), n_bands))
        timesteps = int(((curve_range[1] - curve_range[0]) / interval_val + 1) * n_bands)
        #
        # Declare a dataframe
        #
        data = pd.DataFrame()
        #
        #
        #
        try:
            #
            # Check the type of the file
            #
            if self.type == "transients":
                #
                for i, csv in enumerate(filename):
                    #
                    # Store the IAU Name of each files
                    #
                    id = re.findall("_(.*?)_ZTF\d+[a-zA-Z]{1,10}_processed", csv)
                    #
                    #
                    #
                    df = pd.read_csv(self.path + csv)
                    df.index = df["relative_time"]
                    df = df.fillna(maskval)
                    mwebv_list.append(df["mwebv"].unique())
                    #
                    #
                    #
                    combined_df = pd.DataFrame()
                    #
                    for band in self.passbands:

                        pb_df = df[["relative_time"]].copy()
                        pb_df["uncert"] = df[f"{band}_uncert"]
                        pb_df["flux"] = df[f"{band}_flux"]
                        pb_df[(pb_df["flux"] == maskval) & (pb_df["uncert"] == maskval)] = maskval
                        pb_df["kind"] = band

                        for t in np.arange(curve_range[0], curve_range[1] + interval_val, interval_val):
                            if t not in pb_df["relative_time"]:
                                pb_df.loc[t] = np.full(shape=len(pb_df.columns), fill_value=maskval)
                                pb_df = pb_df.sort_index()

                        max_flux_list.append(max(pb_df["flux"]))
                        data_ = scaler.fit_transform(pb_df[["flux"]])
                        pb_df.flux = data_[:, 0]
                        combined_df = pd.concat([pb_df, combined_df])
                    combined_df["id"] = np.full(shape=timesteps, fill_value=id)
                    data = pd.concat([combined_df, data])
                    max_flux[i:] = np.full(shape=n_bands, fill_value=max_flux_list)
                    max_flux_list.clear()

                data = data.drop(columns="uncert")
                data = data[data["relative_time"] != 0.0]
                data = data.rename(columns={'relative_time': 'time'})
                data = data.reset_index()
                data = data.drop(columns="relative_time")

                columns = [f"{band}_max_flux" for band in self.passbands]

                self.tsfresh_data = data
                self.max_flux = pd.DataFrame(max_flux, columns=columns)
                self.mwebv = pd.DataFrame(mwebv_list, columns=["mwebv"])

        except Exception as e:
            print(e)
            return

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
                extracted_features = tf.extract_features(self.tsfresh_data, column_id='id', column_sort='time',
                                                         column_kind='kind', column_value='flux')

            else:
                raise ValueError(f"\nValueError: Please check the -- type -- of the file!\n"
                                 f"It should be transients/transits type. '{self.type}' is an invalid type!\n")
            #
            # Drop columns and rows that have NaNs
            # Drop any [np.inf, -np.inf] with NaNs and
            # Fill NaNs with a high positive integer
            #
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


    def get_important_features(self, path=None, method="pca"):

        """
        Generates important features from the TSFresh data

        Parameters
        ----------
        path: string
            the file location of the TSFresh extracted features

        method: string (default = pca)
            method to extract 'n_features' using - Linear_PCA (pca) or Unsupervised_RF (urf)

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
            if method not in ["urf", "pca"]:
                raise TypeError(f"\nTypeError: Unknown method type!\nPlease provide method as urf or pca.\n")
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
            exit()

        nunique = tsfresh_data.nunique()
        cols_to_drop = nunique[nunique == 1].index
        tsfresh_data = tsfresh_data.drop(cols_to_drop, axis=1)
        #
        # Extract - n_features - using pca or urf
        #
        try:
            self.labels = tsfresh_data.index.to_list()
            #
            # Convert the dataframe to numpy array
            #
            extracted_features = tsfresh_data.to_numpy()
            #
            #
            #
            feature_list = list()
            feature_imp = dict()
            col = tsfresh_data.columns

            if method == "urf":
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
                   'random_state': 0,
                   'lc_type': self.type,
                   'extract_type': 'tsfresh'
                   }
                #
                # Initialize Unsupervised RF classifier
                #
                URF = UnsupervisedRandomForest(**params)
                #
                # Get the important features
                #
                features = URF.get_feature_importance()

                for i in range(self.n_features):
                    idx, val = features[i][0], features[i][1]
                    feature_list.append(idx)
                    feature_imp[col[idx]] = val
                #
                # Reshape the dataframe
                #
                extracted_df = tsfresh_data.iloc[:, feature_list]
                extracted_df = extracted_df.reset_index(drop=True)
                #
                # Create dictionary to add metadata
                #
                data = {'data': extracted_df, 'labels': self.labels, 'feature_imp': feature_imp}

            elif method == "pca":
                #
                # Initialize linear_PCA classifier
                #
                linear_PCA = PCA(n_components=self.n_features)
                linear_PCA.fit(extracted_features, y=None)
                eigenvectors = linear_PCA.components_
                #
                # Store the important features
                #
                features_score = abs(eigenvectors[0, :])
                features = np.arange(len(features_score))
                features = sorted(zip(features, features_score), key=lambda x: x[1], reverse=True)
                for i in range(self.n_features):
                    idx, val = features[i][0], features[i][1]
                    feature_list.append(idx)
                    feature_imp[col[idx]] = val
                #
                # Reshape the dataframe
                #
                extracted_df = tsfresh_data.iloc[:, feature_list]
                #
                # Create dictionary to add metadata
                #
                data = {'data': extracted_df, 'labels': self.labels, 'feature_imp': feature_imp}
                #
                # Load the Linear PCA transformed data and add metadata to the file
                #
            try:
                #
                # Store the file in -- '/latent_space_data/{type}/' folder
                #
                with open(f"../latent_space_data/{self.type}/tsfresh.pickle", 'wb') as file:
                    pickle.dump(data, file)

            except Exception as e:
                print(f"\nUnknownError: {e}\n")
                return
            #
            #
            #
            print(f"\nTSFresh latent space data is extracted and stored "
                  f"in -- /latent_space_data/{self.type} -- folder!\n")

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return

    def save_data(self, path=None):

        #
        # Load TSFresh data to add metadata
        #
        try:
            with open(path, 'rb') as file:
                tsfresh_data = pickle.load(file)

        except Exception as e:
            print(f"\nFileNotFound: Unable to load the .pickle file!\n")
            exit()

        extracted_df = tsfresh_data["data"]

        try:

            if not self.mwebv.empty and self.metadata and "mwebv" in self.metadata:
                extracted_df = pd.concat([extracted_df.reset_index(drop=True), self.mwebv.reset_index(drop=True)], axis=1)

            if not self.max_flux.empty and self.metadata and "max_flux" in self.metadata:
                extracted_df = pd.concat([extracted_df.reset_index(drop=True), self.max_flux.reset_index(drop=True)], axis=1)

            tsfresh_data["data"] = extracted_df

        except Exception as e:
            print(f"\nUnknownError: {e}\n")

        try:
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(path, 'wb') as file:
                pickle.dump(tsfresh_data, file)
            print(f"\nData stored!\n")

        except Exception as e:
            print(f"\nUnknownError: {e}\n")

        return


if __name__ == '__main__':

    tsfresh = TSFresh(lc_type="transients", path=f"../transients/processed_curves_good_great/",
                      passbands=["r", "tess", "g"], metadata=["mwebv", "max_flux"], n_features=15)
    tsfresh.generate_data()
    tsfresh.extract_features(path=f"../transients/data/tsfresh_data.pickle")
    tsfresh.get_important_features(path="../transients/data/tsfresh_data.pickle", method="urf")
    tsfresh.save_data(path="../latent_space_data/transients/tsfresh.pickle")






