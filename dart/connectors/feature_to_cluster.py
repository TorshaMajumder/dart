#
# Import all the dependencies
#
import pickle
import numpy as np


def combine_features(extract_type=None):
    #
    # Declare all the variables
    # The band names are in the format - g_band, r_band
    #
    band_flux = list()
    data = dict()
    bands = ["g_band", "r_band"]
    #
    # Load the transient names
    #
    try:
        with open(f"../transients/plasticc_labels_.pickle", 'rb') as f:
            load_data = pickle.load(f)
        labels = [k for k in load_data.keys()]
    except Exception as e:
        print(f"\n\nException Raised: {e}\n\n")
    #
    # Concatenate the features of both the bands
    # store it as a numpy array under the
    # folder as - "../latent_space_data/transients/{extract_type}.pickle"
    #
    for band in bands:
        #
        path = f"../latent_space_data/transients/{extract_type}_{band}.pickle"
        try:
            with open(path, 'rb') as f:
                load_data = pickle.load(f)
                if not isinstance(load_data, (np.ndarray, np.generic)):
                    load_data = np.array(load_data)
            band_flux.append(load_data)
        except Exception as e:
            print(f"\n\nException Raised: {e}\n\n")
            return
    #
    #
    #
    flux = np.concatenate((band_flux[0], band_flux[1]), axis=1)
    #
    data["data"] = flux
    data["labels"] = labels
    #
    with open(f"../latent_space_data/transients/{extract_type}.pickle", 'wb') as file:
        pickle.dump(data, file)
    return data


if __name__ == '__main__':

    data = combine_features(extract_type="umap")
