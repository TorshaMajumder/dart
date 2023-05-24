#
# Import all the dependencies
#
import os
import pickle
import numpy as np
import pandas as pd
#
# Create '/transients' folder if it does not exist already
#
if not os.path.exists("../transients"):
    os.makedirs("../transients")
#
# Set all the parameters
#
path_to_store = f"../transients"


def main():
    #
    # Declare a dictionary to store the fluxes
    # with keys as - "g_band" and "r-band"
    #
    flux = dict()
    #
    # Calculate the timesteps for each bands
    #
    curve_range, interval_val = (-30, 70), 3.0
    timesteps = int((curve_range[1] - curve_range[0]) / interval_val + 1)
    #
    # Declare the path to store the fluxes
    #
    # Iterate through all the files in the folder
    # Store the fluxes for each band as a dictionary
    # under the folder /transients
    # The file name is - flux.pickle
    #
    try:
        path = f"{path_to_store}/plasticc_data/"
        filenames = os.listdir(path)

        #
        g_flux = np.zeros((len(filenames), timesteps))
        r_flux = np.zeros((len(filenames), timesteps))
        #
        for i, file in enumerate(filenames):
            data = pd.read_csv(path + file)
            g_flux[i] = data["g_flux"].values.reshape((1, timesteps))
            r_flux[i] = data["r_flux"].values.reshape((1, timesteps))
        #
        #
        #
        g_flux[np.isnan(g_flux)] = 0
        r_flux[np.isnan(r_flux)] = 0
        flux["g_band"] = g_flux
        flux["r_band"] = r_flux
    except Exception as e:
        print(f"\n\nException Raised: {e}\n\n")

    #
    # Store the dictionary as a pickle file
    #
    with open(f"{path_to_store}/flux.pickle", 'wb') as f:
        pickle.dump(flux, f)


if __name__ == '__main__':
    main()



