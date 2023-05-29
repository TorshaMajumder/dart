#
# Import all the dependencies
#
import os
import pickle
import numpy as np
import pandas as pd
from dart.connectors.feature_to_cluster import combine_features


def load_latent_space(extract_type=None):

    #combine_features(extract_type=extract_type)

    file_list = list()

    try:
        if os.path.exists('../latent_space_data/transients'):

            for file in os.listdir('../latent_space_data/transients'):
                file_list.append(file)

            try:
                if extract_type == "k_pca":
                    if f"{extract_type}.pickle" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: '{extract_type}.pickle' file "
                                                f"doesn't exists in the folder - /latent_space_data/transients.\n")

                    else:
                        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as file:
                            data = pickle.load(file)

                elif extract_type == "tsfresh":
                    if f"{extract_type}.pickle" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: '{extract_type}.pickle' file "
                                                f"doesn't exists in the folder - /latent_space_data/transients.\n")

                    else:
                        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as file:
                            data = pickle.load(file)

                elif extract_type == "umap":
                    if f"{extract_type}.pickle" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: '{extract_type}.pickle' file "
                                                f"doesn't exists in the folder - /latent_space_data/transients.\n")

                    else:
                        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as file:
                            data = pickle.load(file)

                elif extract_type == "isomap":
                    if f"{extract_type}.pickle" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: '{extract_type}.pickle' file "
                                                f"doesn't exists in the folder - /latent_space_data/transients.\n")

                    else:
                        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as file:
                            data = pickle.load(file)

                elif extract_type == "vae":
                    if f"{extract_type}.pickle" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: '{extract_type}.pickle' file "
                                                f"doesn't exists in the folder - /latent_space_data/transients.\n")

                    else:
                        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as file:
                            data = pickle.load(file)

                else:
                    raise TypeError(f"\nTypeError: expected 'extract_type' as "
                                    f"k_pca, tsfresh, isomap,or vae, but got {extract_type}.\n")

                if isinstance(data['data'], (np.ndarray, np.generic)):
                    return data

                elif isinstance(data['data'], pd.DataFrame):
                    data['data'] = data['data'].to_numpy()
                    return data


                else:
                    raise TypeError(f"\nTypeError: Unable to load the file. Expected a format as "
                                    f"a numpy array or a pandas dataframe.\n")

            except Exception as e:
                print(e)
                exit()

        else:
            raise FileNotFoundError(f"\nFileNotFoundError: Data cannot be loaded!"
                                    f"\nPlease verify if the folder - /latent_space_data/transients - exists.\n")

    except Exception as e:
        print(e)
        exit()


if __name__ == '__main__':

    data = load_latent_space(extract_type='umap')




