#
# Import all the dependencies
#
import os
import pickle
import pandas as pd
import seaborn as sns
from joypy import joyplot
from matplotlib import colors
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
#
# Set global parameters
#
csfont = {'fontname': 'Comic Sans MS'}
#
# Create a '/images' folder if it does not exists already
#
if not os.path.exists('images'):
    os.makedirs('images')


def generate_data(lc_type='transients', filename=None):
    """
    Generates data to visualize the distribution

    Parameters
    ----------
    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    filename: String
        the file name for the clustering method and extract type.
        The default location is - ../results/clustering/{lc_type}/{filename}

    Returns
    -------
    data: Pandas DataFrame

    """
    #
    # Declare the variables
    #
    c_ttypes_dict = dict()
    #
    #
    #
    try:
        if lc_type not in ["transits", "transients"]:
            raise TypeError(f"\nTypeError: '{lc_type}' is not a valid type!"
                            f"\nPlease provide the type as - 'transits' or 'transients'")
    except Exception as e:
        print(e)
        exit()
    #
    # Load the cluster metadata file
    #
    try:
        with open(f"../results/anomaly_detection/{lc_type}/{filename}", 'rb') as file:
            anomaly_info = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../results/anomaly_detection/{lc_type}/{filename} - exists.\n")
        exit()
    #
    # Load the transient types and sub-types
    #
    try:
        #labels = pd.read_csv("../transients/labels__.csv")
        labels = pd.read_csv("../transients/labels_plasticc__.csv")
        labels = labels.rename(columns={'Label': 'Transient_Sub_Type', 'Group': 'Transient_Type'})

    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../transients/labels__.csv - exists.\n")
        exit()

<<<<<<< HEAD:dart/visualize/anomaly.py
=======
    test = pd.read_csv("../transients/transients.csv")
>>>>>>> main:TESS/visualize/anomaly.py
    #
    # Load the IAU_Name and their labels
    #
    try:
        #with open(f'../transients/data/transient_labels.pickle', 'rb') as file:
        with open(f'../transients/data/transient_labels_plasticc.pickle', 'rb') as file:
            labels_info = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../transients/data/transient_labels.pickle - exists.\n")
        exit()
    try:
        #
        # Load the anomaly metadata file to a pandas dataframe
        #
        data = pd.DataFrame({'IAU_Name': anomaly_info['labels'],
                             'Anomaly_Score': anomaly_info['anomaly_score']})
        #
        # Load the IAU_Name and their labels to a pandas dataframe
        #
        labels_info = pd.DataFrame.from_dict(labels_info, orient='index')
        labels_info = labels_info.reset_index()
        labels_info.columns = ["IAU_Name", "Transient_Sub_Type"]
        #
        # Merge the labels_info to the dataframe - 'data'
        #
        data = data.merge(labels_info, on='IAU_Name', how='inner')
        #
        # Merge the labels to the dataframe - 'data' and drop the column - 'Counts'
        #
        data = data.merge(labels, on='Transient_Sub_Type', how='left')
        data = data.drop(columns="Counts")
        #
        # Calculate the total number of transient type and
        # generate individual colors for transient types
        #
        n_ttypes = len(data['Transient_Type'].unique())
        c_ttypes = mcp.gen_color(cmap='Spectral', n=n_ttypes)
        #
        # Create a dict() of colors for each transient types
        #
        for i, val in enumerate(zip(data.Transient_Type.unique(), c_ttypes)):
            c_ttypes_dict[val[0]] = val[1]
        #
        # Create a dataframe to store the transient type colors
        # and merge it to the dataframe - 'data'
        #
        ttypes_df = pd.DataFrame.from_dict(c_ttypes_dict, orient='index')
        ttypes_df = ttypes_df.reset_index()
        ttypes_df.columns = ["Transient_Type", "Transient_Type_Color"]
        #
        #
        #
        data = data.merge(ttypes_df, on='Transient_Type', how='left')

    except Exception as e:
        print(e)
        exit()

    return data


def visualize_anomaly_with_sns(data=None, extract_type=None, method=None, lc_type="transients"):
    """"
    Generates a plot to visualize the anomalies as a PDF using seaborn

    Parameters
    ----------
    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    data: Pandas DataFrame
        dataframe with the distribution information

    extract_type: String
        feature extraction methods - k_pca, tsfresh, isomap,or vae

    method: String
        anomaly detection methods - iforest, hdbscan, or urf

    Returns
    -------
    Generates images in -- images/{lc_type}/anomaly/ - folder


    """
    #
    # Declare the variables
    #
    n_ttypes = len(data['Transient_Type'].unique())
    urf_xticks = (0.6, 1.0)
    #
    # Create a 'images/{lc_type}/anomaly/' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/anomaly/'):
        os.makedirs(f'images/{lc_type}/anomaly/')
    #
    #
    #
    try:
        if lc_type not in ["transits", "transients"]:
            raise TypeError(f"\nTypeError: '{lc_type}' is not a valid lc_type!"
                            f"\nPlease provide the lc_type as - 'transits' or 'transients'")
    except Exception as e:
        print(e)
        exit()

    try:
        if extract_type not in ["k_pca", "tsfresh", "vae", "isomap"]:
            raise TypeError(f"\nTypeError: '{extract_type}' is not a valid extract_type!"
                            f"\nPlease provide the extract_type as - 'k_pca' , 'tsfresh', 'isomap',or 'vae'")
        else:
            #
            # Generate the labels
            #
            if extract_type == "k_pca":
                label2 = "Kernel PCA"
            elif extract_type == "tsfresh":
                label2 = "TSFresh"
            elif extract_type == "vae":
                label2 = "Variational Auto-Encoder"
            elif extract_type == "isomap":
                label2 = "Isomap"
    except Exception as e:
        print(e)
        exit()
    try:
        if method not in ["iforest", "urf", "hdbscan"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid method!"
                            f"\nPlease provide the method as - 'iforest', hdbscan, or 'urf'")
        else:
            #
            # Generate the labels
            #
            if method == "iforest":
                label1 = "Isolation Forest"
            elif method == "urf":
                label1 = "Unsupervised Random Forest"
            elif method == "hdbscan":
                label1 = "HDBSCAN"
    except Exception as e:
        print(e)
        exit()
    #
    # Generate the plot
    #
    fig, axs = plt.subplots(nrows=n_ttypes, ncols=1, figsize=(15, 17))
    fig.suptitle(f'Anomaly Detection: {label1} ----- Feature Extraction: {label2}', **csfont, fontsize=20)
    #
    # Set the index
    #
    i = 0
    #
    #
    #
    for ttype, d in data.groupby('Transient_Type'):

        plot = sns.distplot(d['Anomaly_Score'], rug=False, hist=True, ax=axs[i],
                            color=d['Transient_Type_Color'].unique()[0], kde=True,
                            kde_kws={"color": "k", "lw": 2, "label": "KDE"})
        # l_ = plot.lines[0]
        # x = l_.get_xydata()[:, 0]
        # y = l_.get_xydata()[:, 1]
        # plot.fill_between(x, y, color=d['Transient_Type_Color'].unique()[0], alpha=0.40)
        plot.set(yticklabels=[])
        plot.set_xlabel("Anomaly Score", fontsize=14)
        plot.set_ylabel(ttype, fontsize=15)
        plot.grid(color='lightgray', linestyle='dotted', zorder=0)
        #
        #
        #
        i += 1
    #
    plt.tight_layout(pad=1.5)
    plt.savefig(f'images/{lc_type}/anomaly/{method}_{extract_type}_sns.png', bbox_inches='tight')

    print(f"\nThe anomaly score plot is generated and stored in - "
          f"images/{lc_type}/anomaly/ - folder!\n")

    return


def visualize_anomaly_with_joypy(data=None, extract_type=None, method=None, lc_type=None):
    """"
    Generates a plot to visualize the anomalies as a PDF using joypy

    Parameters
    ---------
    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    data: Pandas DataFrame
        dataframe with the distribution information

    extract_type: String
        feature extraction methods - k_pca, tsfresh, or vae

    method: String
        anomaly detection methods - iforest , hdbscan, or urf

    Returns
    --------
    Generates images in -- images/{lc_type}/anomaly/ - folder


    """
    #
    # Declare the variables
    #
    cmap = colors.ListedColormap(data['Transient_Type_Color'].unique())
    urf_xticks = (0.80, 1.05)
    iforest_xticks = (-0.4, 0.4)
    hdbscan_xticks = (-0.2, 0.5)
    #
    # Create a 'images/{lc_type}/anomaly/' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/anomaly/'):
        os.makedirs(f'images/{lc_type}/anomaly/')
    #
    #
    #
    try:
        if lc_type not in ["transits", "transients"]:
            raise TypeError(f"\nTypeError: '{lc_type}' is not a valid lc_type!"
                            f"\nPlease provide the lc_type as - 'transits' or 'transients'")
    except Exception as e:
        print(e)
        exit()

    try:
        if extract_type not in ["k_pca", "tsfresh", "vae", "isomap"]:
            raise TypeError(f"\nTypeError: '{extract_type}' is not a valid extract_type!"
                            f"\nPlease provide the extract_type as - 'k_pca' , 'tsfresh', 'isomap',or 'vae'")
        else:
            #
            # Generate the labels
            #
            if extract_type == "k_pca":
                label2 = "Kernel PCA"
            elif extract_type == "tsfresh":
                label2 = "TSFresh"
            elif extract_type == "vae":
                label2 = "Variational Auto-Encoder"
            elif extract_type == "isomap":
                label2 = "Isomap"
    except Exception as e:
        print(e)
        exit()
    try:
        if method not in ["iforest", "urf", "hdbscan"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid method!"
                            f"\nPlease provide the method as - 'iforest', hdbscan, or 'urf'")
        else:
            #
            # Generate the labels
            #
            if method == "iforest":
                label1 = "Isolation Forest"
            elif method == "urf":
                label1 = "Unsupervised Random Forest"
            elif method == "hdbscan":
                label1 = "HDBSCAN"
    except Exception as e:
        print(e)
        exit()
    #
    # Set the x_ticks for urf, hdbscan, and iforest
    #
    if method == "iforest":
        x_ticks = iforest_xticks
    elif method == "hdbscan":
        x_ticks = hdbscan_xticks
    else:
        x_ticks = urf_xticks
    #
    # Plot the images
    #
    joyplot(
        data=data[['Anomaly_Score', 'Transient_Type']],
        by='Transient_Type',
        figsize=(8, 7),
        alpha=0.8,
        x_range=x_ticks,
        colormap=[cmap],
        )
    plt.title(f'Anomaly Detection: {label1} ----- Feature Extraction: {label2}', **csfont, fontsize=12)
    plt.xlabel('Anomaly Score')
    #
    plt.savefig(f'images/{lc_type}/anomaly/{method}_{extract_type}_joypy.png', bbox_inches='tight')

    print(f"\nThe anomaly score plot is generated and stored in - "
          f"images/{lc_type}/anomaly/ - folder!\n")

    return


if __name__ == '__main__':

<<<<<<< HEAD:dart/visualize/anomaly.py
    data_info = generate_data(lc_type="transients", filename="iforest_k_pca.pickle")
    visualize_anomaly_with_joypy(data_info, lc_type="transients", extract_type="k_pca", method="iforest")
    visualize_anomaly_with_sns(data_info, lc_type="transients", extract_type="k_pca", method="iforest")
=======
    data_info = generate_data(lc_type="transients", filename="urf_k_pca.pickle")
    visualize_anomaly_with_joypy(data_info, lc_type="transients", extract_type="k_pca", method="urf")
    visualize_anomaly_with_sns(data_info, lc_type="transients", extract_type="k_pca", method="urf")
>>>>>>> main:TESS/visualize/anomaly.py

