#
# Import all the dependencies
#
import os
import pickle
import random
import pandas as pd
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from TESS.datasets.transients import load_latent_space
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
    m_ttypes_dict = dict()
    c_clusters_dict = dict()
    c_tsubtypes_dict = dict()
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
        if lc_type == "transits":
            raise NotImplementedError(f"\nNotImplementedError: Cannot accept '{lc_type}' as lc_type!"
                                      f"\nPlease provide the lc_type as - 'transients'.\n")
    except Exception as e:
        print(e)
        exit()
    #
    # Load the cluster metadata file
    #
    try:
        with open(f"../results/clustering/{lc_type}/{filename}", 'rb') as file:
            clusters_info = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../results/clustering/{lc_type}/{filename} - exists.\n")
        exit()
    #
    # Load the transient types and sub-types
    #
    try:
        labels = pd.read_csv("../transients/labels__.csv")
        labels = labels.rename(columns={'Label': 'Transient_Sub_Type'})
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../transients/labels__.csv - exists.\n")
        exit()
    #
    # Load the IAU_Name and their labels
    #
    try:
        with open(f'../transients/data/transient_labels.pickle', 'rb') as file:
            labels_info = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../transients/data/transient_labels.pickle - exists.\n")
        exit()
    try:
        #
        # Load the anomaly metadata file to a pandas dataframe
        #
        data = pd.DataFrame.from_dict(clusters_info)
        data = data.rename(columns={'labels': 'IAU_Name', 'clusters': 'Cluster_ID'})
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
        data = data.merge(labels, on='Transient_Sub_Type', how='inner')
        data = data.rename(columns={'Group': 'Transient_Type'})
        #
        # Calculate the number of clusters, transient types and sub-types
        #
        n_clusters = len(data.Cluster_ID.unique())
        n_tsubtypes = len(data.Transient_Sub_Type.unique())
        #
        # Generate colors for different clusters and transient sub-types
        #
        c_clusters = mcp.gen_color(cmap="Set2", n=n_clusters)
        palette = sns.color_palette(cc.glasbey, as_cmap=True)
        c_tsubtypes = random.choices(palette, k=n_tsubtypes)
        #
        # Create a dict() of colors for each clusters
        #
        for i, val in enumerate(zip(data.Cluster_ID.unique(), c_clusters)):
            c_clusters_dict[val[0]] = val[1]
        #
        # Create markers for each transient types
        #
        ttypes_markers = ["*", "X", "o", "^", "P", "D", "s"]
        for i, val in enumerate(zip(data.Transient_Type.unique(), ttypes_markers)):
            m_ttypes_dict[val[0]] = val[1]
        #
        # Create a dataframe for transient type info and merge it to 'data'
        #
        ttype_df = pd.DataFrame.from_dict(m_ttypes_dict, orient='index')
        ttype_df = ttype_df.reset_index()
        ttype_df.columns = ["Transient_Type", "Transient_Type_Marker"]
        #
        #
        #
        data = data.merge(ttype_df, on='Transient_Type', how='inner')
        c_color = [c_clusters_dict[c] for c in data.Cluster_ID.to_list()]
        data["Cluster_Color_ID"] = c_color
        #
        # Create a dict() of colors for each transient subtypes
        #
        for i, val in enumerate(zip(data.Transient_Sub_Type.unique(), c_tsubtypes)):
            c_tsubtypes_dict[val[0]] = val[1]
        #
        # Create a dataframe for transient sub-type info and merge it to 'data'
        #
        tsubtype_df = pd.DataFrame.from_dict(c_tsubtypes_dict, orient='index')
        tsubtype_df = tsubtype_df.reset_index()
        tsubtype_df.columns = ["Transient_Sub_Type", "Transient_Sub_Type_Color"]
        #
        #
        #
        data = data.merge(tsubtype_df, on='Transient_Sub_Type', how='inner')
        #
        # Calculate the total number of transient type and
        # generate individual colors for transient types
        #
        n_ttypes = len(data['Transient_Type'].unique())
        c_ttypes = mcp.gen_color(cmap="Set2", n=n_ttypes)
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
        print(f"\nUnable to generate the data!"
              f"\nException Raised: {e}\n")
        exit()

    print(f"\nData is generated!\n")

    return data


def visualize_dist(data=None, lc_type='transients'):

    """
    Generates a pie plot to visualize the distribution

    Parameters
    ----------
    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    data: Pandas DataFrame
        dataframe with the distribution information

    Returns
    --------
    Generates images in -- images/{lc_type}/distribution/ - folder

    """
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
        if lc_type == "transits":
            raise NotImplementedError(f"\nNotImplementedError: Cannot accept '{lc_type}' as lc_type!"
                                      f"\nPlease provide the lc_type as - 'transients'.\n")
    except Exception as e:
        print(e)
        exit()
    #
    # Create a 'images/{lc_type}/distribution/' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/distribution/'):
        os.makedirs(f'images/{lc_type}/distribution/')
    #
    # Calculate the counts of the transient types and merge it into transients dataframe
    #
    ttypes_count_df = data["Transient_Type"].value_counts().reset_index()
    column = ['Transient_Type','Transient_Type_Counts']
    ttypes_count_df.columns = column
    #
    # Create a dataframe for transient type info
    #
    ttype_df = data[["Transient_Type", "Transient_Type_Color"]]
    ttype_df = ttype_df.drop_duplicates(subset='Transient_Type', keep="first")
    ttype_df = ttype_df.merge(ttypes_count_df, on='Transient_Type', how='inner')
    #
    # Calculate the counts of the transient sub-types and merge it into transients dataframe
    #
    tstypes_count_df = data["Transient_Sub_Type"].value_counts().reset_index()
    column = ['Transient_Sub_Type','Transient_Sub_Type_Counts']
    tstypes_count_df.columns = column
    #
    # Create a dataframe for transient sub-type info
    #
    tstype_df = data[["Transient_Sub_Type", "Transient_Sub_Type_Color"]]
    tstype_df = tstype_df.drop_duplicates(subset='Transient_Sub_Type', keep="first")
    tstype_df = tstype_df.merge(tstypes_count_df, on='Transient_Sub_Type', how='inner')
    #
    # Calculate the percentage of each transient types
    #
    ttype_counts = ttype_df["Transient_Type_Counts"].to_numpy()
    percent = 100.*ttype_counts/ttype_counts.sum()
    explode = (0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    #
    # Plot the pie-plot
    #
    plt.title( f"Data Distribution", **csfont, size=14)
    patches, texts = plt.pie(ttype_df["Transient_Type_Counts"], explode=explode,
                             colors=ttype_df["Transient_Type_Color"],
                             startangle=90, radius=1.1)
    #
    # Generate the labels for the legend
    #
    labels_ = [f'{i}: {k} ({j:1.2f}%)' for i, j, k in zip(ttype_df["Transient_Type"], percent, ttype_counts)]
    patches, labels_, dummy = zip(*sorted(zip(patches, labels_, ttype_df["Transient_Type_Counts"]),
                                          key=lambda x: x[2],
                                          reverse=True))
    #
    # Plot the legend
    #
    plt.legend(patches, labels_, loc='center left', bbox_to_anchor=(-0.4, 1.0), fontsize=8)
    #
    # Save the image
    #
    plt.savefig(f'images/{lc_type}/distribution/ttype_dist.png', bbox_inches='tight')

    print(f"\nThe distribution plot is generated and stored in - "
          f"images/{lc_type}/distribution/ - folder!\n")

    return


if __name__ == '__main__':

    data = load_latent_space(extract_type='vae')
    X_train, labels = data['data'], data['labels']
    data_info = generate_data(lc_type="transients", filename="birch_vae.pickle")
    visualize_dist(data=data_info)