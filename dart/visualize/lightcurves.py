#
# Import all the dependencies
#
import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
#
# Create a '/images' folder if it does not exists already
#
if not os.path.exists('images'):
    os.makedirs('images')

def generate_image(iau_name=None, toi=None, lc_type='transients', filename=None):

    """
    Generates a single image for the IAU Name

    Parameters
    ----------
    iau_name: string
        IAU Name of the transients

    toi: string
        TOI ID of the transits

    filename: pickle file
        pickle file, which contains metadata for the IAU Names in the dictionary
        format, with the keys as  -- [labels, anomaly_index, anomaly_score]

    lc_type: string (default=transients)
        light curves type (transients/transits)

    Returns
    -------
    Generates images in -- /images/{lc_type}/anomaly/lc_objects/

    """

    try:
        if lc_type not in ["transits", "transients"]:
            raise ValueError(f"\n'{lc_type}' is not a valid type!"
                             f"\nPlease provide the type as - 'transits' or 'transients'")
    except Exception as e:
        print(e)
        exit()

    try:
        if lc_type == "transits":
            raise NotImplementedError(f"\nNotImplementedError: Cannot accept lc_type as - transits!\n")
    except Exception as e:
        print(e)
        exit()
    #
    #
    #
    csfont = {'fontname': 'Comic Sans MS'}
    #
    # Load the transient_labels.pickle file
    #
    try:
        with open(f'../{lc_type}/data/transient_labels.pickle', 'rb') as file:
            labels_ = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../{lc_type}/data/transient_labels.pickle - exists.\n")
        exit()
    #
    # Load the anomaly metadata file
    #
    try:
        with open(f"../results/anomaly_detection/{lc_type}/{filename}", 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../results/anomaly_detection/{lc_type}/{filename} - exists.\n")
        exit()
    #
    # Extract the IAU Name and anomaly score and store it in a sorted dictionary
    #
    iau_name_list = [val for i, val in enumerate(data['labels']) if i in data['anomaly_index']]
    anomaly_score = [val for i, val in enumerate(data['anomaly_score']) if i in data['anomaly_index']]
    score_dict = dict(zip(iau_name_list, anomaly_score))
    file = f"{iau_name}.csv.gz"
    #
    # Create a 'images/{lc_type}/lc_objects' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/anomaly/lc_objects'):
        os.makedirs(f'images/{lc_type}/anomaly/lc_objects')
    #
    #
    #
    fig = plt.figure(figsize=(10, 8))
    try:
        class_ = labels_[iau_name]
        score = score_dict[iau_name]
        fig.suptitle(f"IAU Name : {iau_name}  -----  Label : {class_} ----- Anomaly Score : {score:.4f}",
                     fontsize=15, **csfont)
        binned_lc = pd.read_csv(f"../transients/processed_transients/{file}", compression='gzip')
        x = np.array(binned_lc['time'])
        y = np.array(binned_lc['flux'])
        e = np.array(binned_lc['flux_err'])
        plt.scatter(x, y, s=30, c='black')
        plt.grid(color='black', linestyle='-.', linewidth=0.5)
        plt.errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='lightgray',
                     elinewidth=4, capsize=0)
        plt.xlabel('Time', fontsize=12, **csfont)
        plt.ylabel('Flux', fontsize=12, **csfont)
        plt.tick_params(axis='x', labelsize=10)
        plt.tick_params(axis='y', labelsize=10)
        fig.tight_layout(pad=1.0)
        fig.savefig(f"images/{lc_type}/anomaly/lc_objects/{iau_name}.png", bbox_inches="tight", orientation='landscape')

        print(f"\nImage is available in -- /images/{lc_type}/anomaly/lc_objects/ -- !\n")

    except Exception as e:
        print(f"Exception Raised: {e}\n\n"f"Validate transient: {iau_name}")





def generate_anomaly(filename=None, lc_type='transients'):

    """
    Generates images for all the available IAU Name in '../results/anomaly_detection/{lc_type}/{filename}'

    Parameters
    ----------
    filename: pickle file
        pickle file, which contains metadata for the IAU Names in the
        dictionary format, with the keys as  -- [labels, anomaly_index, anomaly_score]

    lc_type: string (default=transients)
        light curves type (transients/transits)

    Returns
    -------
    Generates images in -- /images/{lc_type}/{extract_filename}


    """

    try:
        if lc_type not in ["transits", "transients"]:
            raise ValueError(f"\n'{lc_type}' is not a valid type!"
                             f"\nPlease provide the type as - 'transits' or 'transients'")
    except Exception as e:
        print(e)
        exit()
    #
    #
    #
    csfont = {'fontname': 'Comic Sans MS'}
    extract_filename = re.findall("(.*?).pickle", filename)
    extract_filename = extract_filename[0]

    file_list = list()
    #
    # Load the transient_labels.pickle file
    #
    try:
        with open(f'../{lc_type}/data/transient_labels.pickle', 'rb') as file:
            labels_ = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../{lc_type}/data/transient_labels.pickle - exists.\n")
        exit()
    #
    # Load the anomaly metadata file
    #
    try:
        with open(f"../results/anomaly_detection/{lc_type}/{filename}", 'rb') as file:
            data = pickle.load(file)
        #
        # Create a 'images/{lc_type}/{extract_filename}' folder if it does not exists already
        #
        if not os.path.exists(f'images/{lc_type}/anomaly/{extract_filename}'):
            os.makedirs(f'images/{lc_type}/anomaly/{extract_filename}')

    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../results/anomaly_detection/{lc_type}/{filename} - exists.\n")
        exit()
    #
    # Extract the IAU Name and anomaly score and store it in a sorted dictionary
    #
    iau_name = [val for i, val in enumerate(data['labels']) if i in data['anomaly_index']]
    anomaly_score = [val for i, val in enumerate(data['anomaly_score']) if i in data['anomaly_index']]
    score_dict = dict(zip(iau_name, anomaly_score))
    #
    # Sort the IAU Name wrt its anomaly score (ascending order)
    #
    sorted_score_dict = OrderedDict(sorted(score_dict.items(), key=lambda x: x[1]))
    #
    # Create the file list
    #
    for id_ in list(sorted_score_dict.keys()):
        file_list.append(f"{id_}.csv.gz")
    #
    # Generate the images
    #
    n_row, id = 3, 0
    q, r = divmod(len(file_list), n_row)
    if r != 0: q += 1
    batch_size = q
    #
    #
    #
    for batch in range(batch_size):
        fig = plt.figure(figsize=(1024, 2048))
        fig, axs = plt.subplots(nrows=n_row, ncols=1, figsize=(18,15))

        k = id
        for i in range(n_row):

            try:
                if id < len(file_list):
                    t_id = file_list[id][:-7]
                    class_ = labels_[t_id]
                    score = score_dict[t_id]
                    axs[i].set_title(f"IAU Name : {t_id}  -----  Label : {class_} ----- Anomaly Score : {score:.4f}",
                                     fontsize=25, **csfont)
                    binned_lc = pd.read_csv(f"../transients/processed_transients/{file_list[id]}", compression='gzip')
                    x = np.array(binned_lc['time'])
                    y = np.array(binned_lc['flux'])
                    e = np.array(binned_lc['flux_err'])
                    axs[i].scatter(x, y, s=30, c='black')
                    axs[i].grid(color='black', linestyle='-.', linewidth=0.5)
                    axs[i].errorbar(x, y, yerr=e, fmt='o', color='black', ecolor='lightgray',
                                    elinewidth=4, capsize=0)
                    axs[i].set_xlabel('Time', fontsize=18, **csfont)
                    axs[i].set_ylabel('Flux', fontsize=18, **csfont)
                    axs[i].tick_params(axis='x', labelsize=15)
                    axs[i].tick_params(axis='y', labelsize=15)
                    id += 1

            except Exception as e:
                print(f"Exception Raised: {e}\n\n"f"Validate transient: {t_id}")
                id += 1
                continue

        fig.tight_layout(pad=1.0)
        fig.savefig(f"images/{lc_type}/anomaly/{extract_filename}/image_{k}_{id-1}.png", bbox_inches="tight", orientation='landscape')

    print(f"\nImages are available in /images/{lc_type}/anomaly/{extract_filename}/.\n")


if __name__ == '__main__':

    generate_anomaly(filename='iforest_vae.pickle', lc_type='transients')
    generate_image(lc_type='transients', iau_name='2021ynx', filename='iforest_vae.pickle')


