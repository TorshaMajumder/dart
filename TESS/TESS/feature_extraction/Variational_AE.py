#
# Import all the dependencies
#
import os
import pickle

import numpy as np
import tensorflow as tf
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

    type: string
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
                 latent_dim=10, n_neurons=100, type=None, input_dim=1):

        self.type = type
        self.Encoder = None
        self.Decoder = None
        self.epochs = epochs
        self.X_train = X_train
        self.encoded_data = None
        self.decoded_data = None
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.n_neurons = n_neurons
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_features = self.X_train.shape[1]

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

    def get_encoder(self):

        """
        The encoder of the Variational Auto-Encoder
        """

        input = tf.keras.Input((self.n_features, self.input_dim))

        x = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(input)
        x = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='relu', recurrent_activation='hard_sigmoid')(x)

        z_mean = tf.keras.layers.GRU(self.latent_dim, return_sequences=False, activation='linear')(x)
        z_log_var = tf.keras.layers.GRU(self.latent_dim, return_sequences=False, activation='linear')(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(input, [z_mean, z_log_var, z], name="encoder")

        return encoder

    def get_decoder(self):

        """
        The decoder of the Variational Auto-Encoder
        """

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        repeater = tf.keras.layers.RepeatVector(self.n_features)(latent_inputs)

        x = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(repeater)
        x = tf.keras.layers.GRU(self.n_neurons, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid')(x)

        decoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_dim, activation='tanh'))(x)
        decoder = tf.keras.Model([latent_inputs, ], decoder_output, name="decoder")

        return decoder

    def fit_transform(self):

        """
        Fits and Transforms the data using Variational Auto-Encoder
        """
        try:
            #
            # Generate the encoder and decoder
            #
            self.Encoder = self.get_encoder()
            self.Decoder = self.get_decoder()
            #
            # Pass the encoder and decoder for the training step
            #
            model = GenerateModel(self.Encoder, self.Decoder)
            #
            # Compile and fit the training data set
            #
            model.compile(optimizer=tf.keras.optimizers.Adam())
            model.fit(self.X_train, epochs=self.epochs, batch_size=self.batch_size)
            #
            # Get the encoded - flux and flux_err (output - [z_mean, z_log_var, z])
            #
            encoded_flux, encoded_flux_err, _ = model.encoder(self.X_train)
            #
            # Get the decoded flux
            #
            decoded_data = model.decoder(encoded_flux)
            #
            # Save the encoded and decoded data
            #
            self.encoded_data = encoded_flux.numpy()
            self.decoded_data = decoded_data.numpy()
            #
            #
            #
            print(f"\nData has been transformed by Variational Auto-Encoder!\n")
            #
            # Store the file in -- '/latent_space_data/{type}/' folder
            #
            with open(f"../latent_space_data/{self.type}/vae.pickle", 'wb') as file:
                pickle.dump(self.encoded_data, file)

        except Exception as e:
            print(f"\nException Raised: {e}\n")
            return


class GenerateData(object):

    def __init__(self, type=None, n_filters=1, path=None, columns=None):

        """
        Generates data for Variational Auto-Encoder

        Parameters
        ----------
        type: string
            type of light curves (transits or transients)

        n_filters: int (default =1)
            number of filters for the TESS light curves

        path: string
            the file location of the light curves

        """

        self.type = type
        self.path = path
        self.labels = None
        self.n_filters = n_filters
        self.columns = columns

        try:
            if self.type not in ["transits", "transients"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

        try:
            for i in self.columns:
                if i not in ["flux", "flux_err", "time_phase"]:
                    raise ValueError(f"\n'{i}' is not a valid columns!"
                                     f"\nPlease provide columns as - 'flux', 'flux_err', 'time_phase'")
        except Exception as e:
            print(e)
            exit()

    def read_data(self):
        #
        # Load the pickle file
        #
        try:
            with open(self.path, 'rb') as file:
                lightcurves = pickle.load(file)
        except Exception as e:
            print(f"\nFileNotFound: Unable to load the .pickle file!\n")
            exit()
        #
        #
        #
        try:
            flux = lightcurves['flux']
            flux_err = lightcurves['flux_err']
            self.labels = lightcurves['metadata']
            #
            # Check the type of the file
            #
            if self.type == "transients":
                time = lightcurves['time']

                return time, flux, flux_err

            elif self.type == "transits":
                phase = lightcurves['phase']

                return phase, flux, flux_err


            else:
                raise ValueError(f"\nValueError: Please specify the -- type -- of the file!\n"
                                 f"It should be transients/transits type. '{self.type}' is an invalid type!\n")

        except Exception as e:
            print(e)
            return

    def extract_data(self):
        #
        # Generate data for filters
        #
        try:
            if self.n_filters == 1:

                time_phase, flux, flux_err = self.read_data()

                if len(self.columns) == 1:
                    if self.columns[0] != "flux":
                        raise ValueError(f"\n'flux' column is mandatory to train the VAE. "
                                         f"Please provide 'flux' as the column.\n")
                    else:
                        flux = flux.reshape((flux.shape[0], flux.shape[1], 1))
                        print(f"\nData is extracted!\n")
                        return flux, self.labels, self.n_filters*1

                elif len(self.columns) == 2:
                    if "flux" not in self.columns:
                        raise ValueError(f"\n'flux' column is mandatory to train the VAE. "
                                         f"Please provide 'flux' as one of the columns.\n")
                    else:
                        flux = flux.reshape((flux.shape[0], flux.shape[1], 1))

                        if "time_phase" in self.columns:

                            time_phase = np.tile(time_phase, flux.shape[0])
                            time_phase = time_phase.reshape((flux.shape[0], flux.shape[1], 1))
                            data = np.concatenate((flux, time_phase), axis=2)
                            data = data.reshape((flux.shape[0], flux.shape[1], 2))
                            print(f"\nData is extracted!\n")
                            return data, self.labels, self.n_filters*1+1

                        elif "flux_err" in self.columns:

                            flux_err = flux_err.reshape((flux_err.shape[0], flux_err.shape[1], 1))
                            data = np.concatenate((flux, flux_err), axis=2)
                            data = data.reshape((flux.shape[0], flux.shape[1], 2))
                            print(f"\nData is extracted!\n")
                            return data, self.labels, self.n_filters*2

                elif len(self.columns) == 3:

                    flux = flux.reshape((flux.shape[0], flux.shape[1], 1))
                    flux_err = flux_err.reshape((flux_err.shape[0], flux_err.shape[1], 1))
                    time_phase = np.tile(time_phase, flux.shape[0])
                    time_phase = time_phase.reshape((flux.shape[0], flux.shape[1], 1))
                    data = np.concatenate((flux, flux_err, time_phase), axis=2)
                    data = data.reshape((flux.shape[0], flux.shape[1], 3))
                    print(f"\nData is extracted!\n")
                    return data, self.labels, self.n_filters*2+1

                else:
                    raise ValueError(f"\nColumns cannot be more than three."
                                     f"\nPlease provide columns as - 'flux', 'flux_err', 'time_phase'\n")

            elif self.n_filters > 1:
                raise NotImplementedError(f"\nNotImplementedError: Unable to accept more than one filter.\n")

        except Exception as e:
            print(e)
            exit()

    def save_data(self):
        #
        # Load the VAE transformed data and add metadata to the file
        #
        try:
            with open(f"../latent_space_data/transients/vae.pickle", 'rb') as file:
                data_ = pickle.load(file)
            data = {'data':data_, 'labels':self.labels}
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


class GenerateModel(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(GenerateModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == '__main__':

    data = GenerateData(type="transients", path="../transients/data/transients.pickle",
                        n_filters=1, columns=['flux', 'flux_err'])
    X_train, labels, input_shape = data.extract_data()
    vae = VariationalAutoEncoder(X_train=X_train, epochs=2, batch_size=50, latent_dim=10,
                                 type="transients", input_dim=input_shape)
    vae.fit_transform()
    data.save_data()





