#
# Import all the dependencies
#
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.framework.ops import disable_eager_execution


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(
    config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

disable_eager_execution()


#
# Create '/latent_space_data' folder if it does not exists already
#
if not os.path.exists('../latent_space_data'):
    os.makedirs('../latent_space_data')


class VariationalAutoEncoder(object):

    """
    VariationalAutoEncoder is a feature extraction tool for dimensionality reduction
    of the light curves from NASA's TESS telescope.

    The light curves are available in the folder -
        -- ../transients/data/transients.pickle
        -- ../transits/data/transits.pickle


    Parameters
    ----------

    X_train: numpy ndarray
        training data set

    lc_type: string
        type of light curves (transits or transients)

    latent_dim: int (default = 10)
        number of features in the latent space

    epochs: int (default = 100)
        number of epochs to train the model

    batch_size: int (default = None)
        number of samples per gradient update

    n_filters: int (default =1)
        number of filters for the TESS light curves

    n_neurons: int (default = 100)
        the number of hidden units (dimensionality of the output space)


    """

    def __init__(self, X_train=None, epochs=100, batch_size=None, n_filters=1,
                 latent_dim=10, n_neurons=100, lc_type=None, time_id_index=None, flux_index=None):

        self.type = lc_type
        self.epochs = epochs
        self.X_train = X_train
        self.encoded_data = None
        self.decoded_data = None
        self.timesteps = self.X_train.shape[1]
        self.n_filters = n_filters
        self.n_neurons = n_neurons
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_features = self.X_train.shape[2]
        self.time_id_index = time_id_index
        self.flux_index = flux_index

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

    def build_model(self):
        """
        Builds entire RVAE model connected as one model

        Returns:
            model: full rvae model
            encoder: just encoder from model (used for testing and classification)
        """
        maskval = 0.0
        # BUILD ENCODER
        encoder_input = tf.keras.Input((self.timesteps, self.n_features))

        mask = tf.keras.layers.Masking(mask_value=maskval)

        mask_compute = mask(encoder_input)

        layer1 = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', name='gru1_encoder')(mask_compute)

        mask_output = mask.compute_mask(layer1)

        layer2 = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='relu', recurrent_activation='hard_sigmoid', name='gru2_encoder')(layer1)

        z_mean = tf.keras.layers.GRU(self.latent_dim, return_sequences=False, activation='linear', name='gru_latent_mean')(layer2)
        z_log_var = tf.keras.layers.GRU(self.latent_dim, return_sequences=False, activation='linear', name='gru_latent_var')(layer2)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

        # BUILD DECODER
        repeater = tf.keras.layers.RepeatVector(self.timesteps)(z)

        # time and filter id vals
        time_filter = tf.keras.Input(shape=(self.timesteps, 2))

        # concat timestep back
        concat = tf.keras.layers.concatenate((repeater, time_filter), axis=-1)

        # add mask from original input
        concat._keras_mask = mask_output

        layer3 = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', name='gru1_decoder')(concat)

        layer4 = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', name='gru2_decoder')(layer3)

        decoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh', input_shape=(None, 1)))(layer4)


        #BUILD MODEL
        decoder = tf.keras.Model([encoder_input, time_filter], decoder_output)
        decoder.summary()

        # define loss
        vae_loss = self.vae_loss(z_mean, z_log_var)

        # compile model

        decoder.compile(loss=vae_loss, optimizer=tf.keras.optimizers.Adam())

        return encoder, decoder


    def fit_transform(self):
        """
        Fits and Transforms the data using Variational Auto-Encoder
        """
        try:

            time_filter = self.X_train[:, :, self.time_id_index]
            assert (time_filter.shape == (
                self.X_train.shape[0], self.X_train.shape[1], 2))
            Y_train = self.X_train[:, :, [self.flux_index]]

        except Exception as e:
            print(f"\nUnknown Error: {e}\n")

        try:

            encoder, decoder = self.build_model()
            history = decoder.fit([self.X_train, time_filter], Y_train, epochs=self.epochs, batch_size=self.batch_size)
            #
            # Get the encoded - flux and flux_err (output - [z_mean, z_log_var, z])
            #
            encoded_flux, encoded_flux_err, _ = encoder(self.X_train)
            #
            # Get the decoded flux
            #
            decoded_flux = decoder([self.X_train, time_filter])
            #
            # Save the encoded and decoded data
            #
            self.encoded_data = encoded_flux.eval()
            self.decoded_data = decoded_flux.eval()

        except Exception as e:
            print(f"\nUnknown Error: {e}\n")
        #
        #
        #
        print(f"\nData has been transformed by Variational Auto-Encoder!\n")
        #
        # Store the file in -- '/latent_space_data/{type}/' folder
        #
        with open(f"../latent_space_data/{self.type}/vae.pickle", 'wb') as file:
            pickle.dump(self.encoded_data, file)




    def vae_loss(self, encoded_mean, encoded_log_sigma):
        """
        Defines the reconstruction + KL loss in a format acceptable by the Keras model
        """

        kl_loss = - 0.5 * K.mean(1 + K.flatten(encoded_log_sigma) -
                                 K.square(K.flatten(encoded_mean)) - K.exp(K.flatten(encoded_log_sigma)), axis=-1)

        def lossFunction(yTrue,yPred):
            reconstruction_loss = K.log(K.mean(K.square(yTrue - yPred)))
            return reconstruction_loss + kl_loss

        return lossFunction


class GenerateData(object):

    def __init__(self, lc_type=None, passbands=["tess"], path=None, metadata=None):

        """
        Generates data for Variational Auto-Encoder

        Parameters
        ----------
        lc_type: string
            type of light curves (transits or transients)

        n_filters: int (default =1)
            number of filters for the TESS light curves

        path: string
            the file location of the light curves

        """

        self.type = lc_type
        self.path = path
        self.labels = None
        self.passbands = passbands
        self.metadata = metadata

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

        scaler = MinMaxScaler()
        curve_range = (-30, 70)
        label, columns = list(), list()
        filename = os.listdir(self.path)
        col_to_drop = ["max_flux", "mwebv"]
        passbands_metadata = {"tess": 7.865, "r": 6.215, "g": 4.716}
        maskval, interval_val, n_bands = 0.0, 0.5, len(self.passbands)
        timesteps = int(((curve_range[1] - curve_range[0]) / interval_val + 1) * n_bands)

        if self.metadata:
            n_cols = 4 + len(self.metadata)
            col_to_drop = list(set(col_to_drop) - set(self.metadata))
        else:
            n_cols = 4

        passband_values = {i: passbands_metadata[i] for i in self.passbands}

        x_train = np.zeros(shape=(len(filename), timesteps, n_cols))

        for i, csv in enumerate(filename):

            id = re.findall("_(.*?)_ZTF\d+[a-zA-Z]{1,10}_processed", csv)
            label.append(id)
            df = pd.read_csv(self.path + csv)
            df.index = df["relative_time"]
            df = df.fillna(maskval)

            combined_df = pd.DataFrame()

            for pb, id in passband_values.items():

                pb_df = df[["relative_time", "mwebv"]].copy()
                pb_df["uncert"] = df[f"{pb}_uncert"]
                pb_df["flux"] = df[f"{pb}_flux"]
                pb_df["id"] = id
                pb_df[(pb_df["flux"] == maskval) & (pb_df["uncert"] == maskval)] = maskval

                for t in np.arange(curve_range[0], curve_range[1] + interval_val, interval_val):
                    if t not in pb_df["relative_time"]:
                        pb_df.loc[t] = np.full(shape=len(pb_df.columns), fill_value=maskval)
                        pb_df = pb_df.sort_index()
                pb_df["max_flux"] = np.full(shape=len(pb_df), fill_value=max(pb_df["flux"]))
                data_ = scaler.fit_transform(pb_df[["flux", "uncert"]])
                pb_df.flux = data_[:, 0]
                pb_df.uncert = data_[:, 1]
                pb_df[(pb_df["relative_time"] == maskval)] = maskval
                combined_df = pd.concat([pb_df, combined_df])

            if col_to_drop:
                combined_df = combined_df.drop(columns=col_to_drop)

            x_train[i] = combined_df.sort_index().to_numpy()

        query_cols = ['relative_time', 'id']
        cols_index = [combined_df.columns.get_loc(col) for col in query_cols]
        flux_index = combined_df.columns.get_loc("flux")
        self.labels = label

        return x_train, cols_index, flux_index

    def save_data(self):
        #
        # Load the VAE transformed data and add metadata to the file
        #
        try:
            with open(f"../latent_space_data/transients/vae.pickle", 'rb') as file:
                data_ = pickle.load(file)
            data = {'data': data_, 'labels': self.labels}
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(f"../latent_space_data/{self.type}/vae.pickle", 'wb') as file:
                pickle.dump(data, file)

        except Exception as e:
            print(f"\nUnknownError: {e}\n")
            return
        #
        #
        #
        print(f"\nVAE latent space data is extracted and stored "
              f"in -- /latent_space_data/{self.type} -- folder!\n")


class Sampling(tf.keras.layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


if __name__ == '__main__':

    data = GenerateData(lc_type="transients", path=f"../transients/processed_curves_good_great/",
                        passbands=["tess", "g", "r"], metadata=["mwebv", "max_flux"])
    X_train, time_id_index, flux_index = data.generate_data()
    vae = VariationalAutoEncoder(X_train=X_train, epochs=10, batch_size=50, latent_dim=8,
                                 lc_type="transients", time_id_index=time_id_index, flux_index=flux_index)
    vae.fit_transform()
    data.save_data()



