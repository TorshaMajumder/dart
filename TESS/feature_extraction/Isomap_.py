#
# Import all the dependencies
#
import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


#from TESS.transients.read_transients import TESS_Transients
#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class IsoMap(object):
    """

    KernelPCA is a feature extraction tool for dimensionality reduction
    of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle

    Parameters
    ----------

    X_train: numpy ndarray (default = None)
        training data set

    lc_type: string
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

    def __init__(self, X_train=None, lc_type=None, n_features=10, n_jobs=-1, passbands=["tess"],
                 path=None, metadata=None):

        self.X_train = X_train
        self.type = lc_type
        self.n_features = n_features
        self.Isomap_Estimator = None
        self.Isomap_decoder = None
        self.n_jobs = n_jobs
        self.labels = None
        self.path = path
        self.passbands = passbands
        self.metadata = metadata
        self.isomap_data = None

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

        try:

            for i in self.passbands:
                if i not in ["tess", "r", "g"]:
                    raise ValueError(f"\nValueError: '{i}' is an invalid passband!"
                                     f"\nPlease provide passbands as - 'tess', 'r' for ZTF r-band, 'g' for ZTF g-band.")
        except Exception as e:
            print(e)
            exit()

    def binned_transients(self, df=None, interval="3D", time_col="relative_time", uncert="tess_uncert"):
        """
        Generate binned transients

        Parameters
        ----------
        df: pandas-dataframe
            dataframe with columns relative_time, cts and e_cts
        interval: string (default = "0.5D")
            scalar + unit, eg. 0.5D, Units: Day: D, Minute: T, Second: S
        time_col: string (default = "relative_time")
            time column name
        uncert: string (default = "e_cts")
            uncertainty column name


        Returns
        -------
        binned_data: pandas-dataframe
            binned dataframe

        """

        binned_data = df.copy()
        #
        # square e_cts to get variances
        #
        binned_data[uncert] = np.power(binned_data[uncert], 2)
        #
        # bin and find avg variance and avg mean per bin
        #
        binned_data.index = pd.TimedeltaIndex(df[time_col], unit="D").round(interval)
        binned_data = binned_data.resample(interval, origin="start").mean()
        binned_data[time_col] = binned_data.index / pd.to_timedelta('3D')
        binned_data.index = binned_data.index / pd.to_timedelta(interval)
        #
        # sqrt avg vars to get uncertainty in stds
        #
        binned_data[uncert] = np.power(binned_data[uncert], 0.5)


        return binned_data


    def generate_data(self):

        filename = os.listdir(self.path)
        label = list()

        maskval = 0.0
        interval_val_tess = 0.5
        interval_val_rg = 3.0
        curve_range = (-30, 70)
        n_bands = len(self.passbands)
        timesteps_tess = int((curve_range[1] - curve_range[0]) / interval_val_tess + 1)
        timesteps_rg = int((curve_range[1] - curve_range[0]) / interval_val_rg + 1)
        tess_flux = np.zeros(shape=(len(filename), timesteps_tess))
        r_flux = np.zeros(shape=(len(filename), timesteps_rg))
        g_flux = np.zeros(shape=(len(filename), timesteps_rg))
        max_flux = np.zeros(shape=(len(filename), n_bands))
        mwebv = np.zeros(shape=(len(filename), 1))
        scaler = MinMaxScaler()
        g_max, g_min = -99.0, 99.0


        for i, csv in enumerate(filename):
            try:
                id = re.findall("_(.*?)_ZTF\d+[a-zA-Z]{1,10}_processed", csv)
                label.append(id[0])
                data = pd.read_csv(self.path + csv)
                mwebv[i] = data[f"mwebv"].unique()
            except Exception as e:
                print("Unknown Error")

            try:
                for j, band in enumerate(self.passbands):
                    band_data = data[["relative_time", f"{band}_flux", f"{band}_uncert"]].copy(deep=True)
                    band_data.index = band_data["relative_time"]
                    try:
                        if band == "tess":
                            filtered_band_data = band_data[(band_data[f"{band}_flux"].notnull()) &
                                                           (band_data[f"{band}_uncert"].notnull())]
                            nan_count = band_data.loc[:, f"{band}_flux"].isna().sum()
                            if nan_count != timesteps_tess:
                                max_rel_time, min_rel_time = filtered_band_data["relative_time"].max(), \
                                                             filtered_band_data["relative_time"].min()
                                if max_rel_time > g_max:
                                    g_max = max_rel_time
                                if min_rel_time < g_min:
                                    g_min = min_rel_time

                                max_flux[i, j] = max(filtered_band_data.loc[:, f"{band}_flux"])

                                #
                                # Apply MinMaxScalar (feature range = [0,1])
                                #
                                data_ = scaler.fit_transform(band_data[[f"{band}_flux", f"{band}_uncert"]])
                                band_data.loc[:, f"{band}_flux"] = data_[:, 0]
                                band_data.loc[:, f"{band}_uncert"] = data_[:, 1]
                                flux_mean = np.nanmean(data_[:, 0])
                                flux_uncert_mean = np.nanmean(data_[:, 1])
                                time = band_data["relative_time"].values.tolist()

                                for t in np.arange(curve_range[0], curve_range[1] + interval_val_tess, interval_val_tess):

                                    if t not in time:
                                        band_data.loc[t] = np.full(shape=len(band_data.columns),
                                                                   fill_value=[t, maskval, maskval])
                                        band_data = band_data.sort_index()


                                for t in np.arange(min_rel_time, max_rel_time + interval_val_tess, interval_val_tess):

                                    band_data.loc[t, f"{band}_flux"] = flux_mean
                                    band_data.loc[t, f"{band}_uncert"] = flux_uncert_mean

                            band_data = band_data.fillna(0.0)
                            band_data = band_data.sort_index()
                            #
                            # Reshape the dataframe to (60,3)
                            #
                            if (len(band_data) != timesteps_tess):
                                n_row = band_data.shape[0]
                                diff = timesteps_tess - n_row
                                arr = np.zeros((diff, band_data.shape[1]))
                                band_data = band_data.append(pd.DataFrame(arr, columns=band_data.columns), ignore_index=True)

                            tess_flux[i] = band_data[f"{band}_flux"].values

                        elif band == "r" or "g":

                            band_data = data[["relative_time", f"{band}_flux", f"{band}_uncert"]].copy(deep=True)
                            binned_data = self.binned_transients(df=band_data, interval="3D", time_col="relative_time",
                                                                 uncert=f"{band}_uncert")

                            nan_count = binned_data.loc[:, f"{band}_flux"].isna().sum()
                            if nan_count != timesteps_rg:

                                max_rel_time, min_rel_time = binned_data["relative_time"].max(), \
                                                             binned_data["relative_time"].min()
                                time = binned_data["relative_time"].values.tolist()

                                max_flux[i, j] = max(binned_data.loc[:, f"{band}_flux"])

                                for t in np.arange(min_rel_time, max_rel_time, 1):

                                    if t not in time:
                                        binned_data.loc[t] = np.full(shape=len(binned_data.columns),
                                                                     fill_value=[t, maskval, maskval])
                                        binned_data = binned_data.sort_index()
                                #
                                # Apply MinMaxScalar (feature range = [0,1])
                                #
                                data_ = scaler.fit_transform(binned_data[[f"{band}_flux", f"{band}_uncert"]])
                                binned_data.loc[:, f"{band}_flux"] = data_[:, 0]
                                binned_data.loc[:, f"{band}_uncert"] = data_[:, 1]
                                flux_mean = np.nanmean(data_[:, 0])
                                flux_uncert_mean = np.nanmean(data_[:, 1])
                                binned_data.loc[:, f"{band}_flux"].fillna(value=flux_mean, inplace=True)
                                binned_data.loc[:, f"{band}_uncert"].fillna(value=flux_uncert_mean, inplace=True)
                                #
                                # Reshape the dataframe to (60,3)
                                #
                                if (len(binned_data) != timesteps_rg):
                                    n_row = binned_data.shape[0]
                                    diff = timesteps_rg - n_row
                                    arr = np.zeros((diff, binned_data.shape[1]))
                                    binned_data = binned_data.append(pd.DataFrame(arr, columns=binned_data.columns),
                                                                     ignore_index=True)

                                if band == "r":
                                    r_flux[i] = binned_data[f"{band}_flux"].values

                                elif band == "g":
                                    g_flux[i] = binned_data[f"{band}_flux"].values

                    except Exception as e:
                        print(f"Unknown Error: {e}")

            except Exception as e:
                print(f"Unknown Error: {e}")


        x_train = dict()
        x_train["tess_flux"] = tess_flux
        x_train["r_flux"] = r_flux
        x_train["g_flux"] = g_flux
        x_train["max_flux"] = max_flux
        x_train["mwebv"] = mwebv
        self.labels = label
        self.X_train = x_train


    def fit_transform(self, x_train):

        """
        Fits the data to Kernel PCA estimator

        """

        try:

            self.Isomap_Estimator = Isomap(n_components=self.n_features, n_jobs=self.n_jobs)

            transformed_data = self.Isomap_Estimator.fit_transform(x_train, y=None)

            return transformed_data

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return



    def build_model(self):

        transformed_data, decoded_data = dict(), dict()
        isomap_data = pd.DataFrame()

        feature_list = np.zeros(shape=len(self.passbands))
        default_band_order = {0: "tess", 1: "r", 2: "g"}
        band_order = dict()
        count = 0

        try:
            if len(self.passbands) != len(band_order):
                for i in default_band_order.keys():
                    if default_band_order[i] in self.passbands:
                        band_order[count] = default_band_order[i]
                        count += 1
            else:
                band_order = default_band_order
        except Exception as e:
            print(f"\nUnknown Error: {e}\n")

        try:
            q, r = divmod(self.n_features, len(self.passbands))
            for i in range(len(self.passbands)):
                feature_list[i] = q
            if r != 0.0:
                for i in range(r):
                    feature_list[i] += 1

        except Exception as e:
            print(f"\nUnknown Error: {e}\n")

        try:
            for band in self.passbands:
                x_train = self.X_train[f"{band}_flux"]
                data= self.fit_transform(x_train=x_train)
                transformed_data[f"{band}_flux"] = data
                #decoded_data[f"{band}_flux"] = dec_data

            for i in range(len(feature_list)):
                features = int(feature_list[i])
                band = band_order[i]
                isomap_data = pd.concat([pd.DataFrame(transformed_data[f"{band}_flux"][:, 0:features]), isomap_data], axis=1)

        except Exception as e:
            print(f"\nUnknown Error: {e}\n")

        try:

            if "mwebv" in self.metadata:
                isomap_data = pd.concat([pd.DataFrame(self.X_train["mwebv"]), isomap_data], axis=1)

            if "max_flux" in self.metadata:
                isomap_data = pd.concat([pd.DataFrame(self.X_train["max_flux"]), isomap_data], axis=1)

        except Exception as e:
            print(f"\nUnknown Error: {e}\n")

        isomap_data = isomap_data.fillna(0.0)
        isomap_data = isomap_data.to_numpy()

        self.isomap_data = isomap_data


    def save_data(self):

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


        # Create dictionary to add metadata
        #
        data = {'data': self.isomap_data, 'labels':self.labels}
        #
        # Store the file in -- '/latent_space_data/{type}/' folder
        #
        with open(f"../latent_space_data/{self.type}/isomap.pickle", 'wb') as file:
            pickle.dump(data, file)
        #
        #
        #
        print(f"\nKernel_PCA latent space data is extracted and stored "
              f"in -- /latent_space_data/{self.type} -- folder!\n")



if __name__ == '__main__':

    i_map = IsoMap(lc_type="transients", path=f"../transients/processed_curves_good_great/",
                       passbands=["r", "g"], metadata=["mwebv", "max_flux"], n_features=8)
    i_map.generate_data()
    i_map.build_model()
    i_map.save_data()


