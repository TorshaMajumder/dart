#
# Import all the dependencies
#
import os
import shutil
import pickle
import random
import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from scipy.spatial import ConvexHull
from mycolorpy import colorlist as mcp
from TESS.feature_extraction import UMAP_
from TESS.feature_extraction import TSNE_
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


def generate_data(X=None, lc_type='transients', filename=None, image_path=None,
                  method=None, viz_method=None, lc_labels=None):

    """"
    Generates a dataframe to visualize the clusters

    Parameters
    ----------
    X: ndarray
        training data

    filename: String
        the file name for the clustering method and extract type.
        The default location is - ../results/clustering/{lc_type}/{filename}

    image_path: String
        path of the processed images, to store the images in individual cluster id folders

    method: String
        clustering methods - birch or hdbscan

    viz_method: String
        visualization methods - umap or tsne

    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    lc_labels: list
        IAU Name for the data samples

    Returns
    -------
    data: Pandas DataFrame

    """
    #
    # Declare the variables
    #
    file_list = list()
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
    try:
        if viz_method not in ["umap", "tsne"]:
            raise TypeError(f"\nTypeError: '{viz_method}' is not a valid viz_method!"
                            f"\nPlease provide the viz_method as - 'umap' or 'tsne'")
    except Exception as e:
        print(e)
        exit()
    try:
        if method not in ["birch", "hdbscan"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid method!"
                            f"\nPlease provide the method as - 'birch' or 'hdbscan'")
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
    # Load the images from the location - image_path
    #
    try:
        for file in os.listdir(image_path):
            file_list.append(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Images cannot be loaded!"
              f"\nPlease verify if the folder - {image_path} - exists.\n")
        exit()
    #
    # Create a 'images/{lc_type}/clustering' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/clustering/{method}/'):
        os.makedirs(f'images/{lc_type}/clustering/{method}/')
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
        data = data.drop(columns=["Counts"])
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
        ttypes_markers = ["*","X","o","^","P","D","s"]
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
        # Create 2D features using TSNE or UMAP
        #
        if viz_method == "umap":
            umap_ = UMAP_.UMap(X_train=X, lc_type=lc_type, n_features=2)
            transformed_data = umap_.fit_transform()
        elif viz_method == "tsne":
            tsne_ = TSNE_.TSNe_(X_train=X, lc_type=lc_type, n_features=2)
            transformed_data = tsne_.fit_transform()
        #
        #
        #
        feature_ext_df = pd.DataFrame(transformed_data, columns=["Feature#1", "Feature#2"])
        feature_ext_df['IAU_Name'] = lc_labels
        #
        # Merge the extracted features to the dataframe - 'data'
        #
        data = data.merge(feature_ext_df, on='IAU_Name', how='inner')
        #
        # Create a 'images/{lc_type}/' folder if it does not exists already
        #
        if not os.path.exists(f'images/{lc_type}/'):
            os.makedirs(f'images/{lc_type}/')

    except Exception as e:
        print(f"\nUnable to generate the data!"
              f"\nException Raised: {e}\n")
        exit()

    print(f"\nData is generated!\n")
    #
    # Store the images of the IAU_Name for individual clusters
    #
    try:
        for i in data.Cluster_ID.unique():
            #
            # Create a 'images/{lc_type}/clustering/cluster{i}' folder if it does not exists already
            #
            if not os.path.exists(f'images/{lc_type}/{method}/clustering/cluster_{i}'):
                os.makedirs(f'images/{lc_type}/clustering/{method}/cluster_{i}')
            #
            # Store the IAU_Name for individual clusters
            #
            ids = data[data.Cluster_ID == i][["IAU_Name"]].values.tolist()
            #
            # Copy the images to - images/{lc_type}/clustering/cluster_{i}
            #
            for id_ in range(len(ids)):
                file = ids[id_][0]+'.png'
                if file in file_list:
                    shutil.copy(f"processed/{file}", f"images/{lc_type}/clustering/{method}/cluster_{i}")
    except Exception as e:
        print(f"\nUnable to store the cluster images!"
              f"\nException Raised: {e}\n")

    print(f"\nImages are stored in - images/{lc_type}/clustering/ - folder!\n")

    return data


def visualize_clusters(data=None, convex_hull=True, viz_method="umap", method=None,
                       extract_type=None, lc_type='transients'):
    """"
    Generates a plot to visualize the clusters

    Parameters
    ----------
    data: Pandas DataFrame
        dataframe with the distribution information

    convex_hull: Boolean
        plots the convex hull if TRUE

    extract_type: String
        feature extraction methods - k_pca, tsfresh, or vae

    method: String
        clustering methods - birch or hdbscan

    viz_method: String
        visualization methods - umap or tsne

    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'


    Returns
    -------
    Generates images in -- images/{lc_type}/clustering/ - folder

    """
    #
    # Declare the variables
    #
    m_ttypes_dict = dict()
    c_clusters_dict = dict()
    #
    # Create a 'images/{lc_type}/clustering' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/clustering/'):
        os.makedirs(f'images/{lc_type}/clustering/')
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
    try:
        if extract_type not in ["k_pca", "tsfresh", "vae"]:
            raise TypeError(f"\nTypeError: '{extract_type}' is not a valid extract_type!"
                            f"\nPlease provide the extract_type as - 'k_pca' , 'tsfresh', or 'vae'")
        else:
            #
            # Generate the labels
            #
            if extract_type == "k_pca":
                label1 = "Kernel PCA"
            elif extract_type == "tsfresh":
                label1 = "TSFresh"
            elif extract_type == "vae":
                label1 = "Variational Auto-Encoder"
    except Exception as e:
        print(e)
        exit()
    try:
        if method not in ["birch", "hdbscan"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid method!"
                            f"\nPlease provide the method as - 'birch' or 'hdbscan'")
        else:
            #
            # Generate the labels
            #
            if method == "birch":
                label2 = "Birch"
            elif method == "hdbscan":
                label2 = "HDBSCAN"
    except Exception as e:
        print(e)
        exit()
    try:
        if viz_method not in ["umap", "tsne"]:
            raise TypeError(f"\nTypeError: '{viz_method}' is not a valid viz_method!"
                            f"\nPlease provide the viz_method as - 'umap' or 'tsne'")
        else:
            #
            # Generate the labels
            #
            if viz_method == "umap":
                label3 = "UMAP"
            elif viz_method == "tsne":
                label3 = "TSNE"
    except Exception as e:
        print(e)
        exit()
    #
    # Create a dataframe to count the total transients in each cluster
    #
    cluster_count = data.groupby(["Cluster_ID"]).count()
    cluster_count = cluster_count.reset_index()
    cluster_count = cluster_count.drop(columns=["Feature#1", "Feature#2", "IAU_Name", "Transient_Sub_Type"])
    cluster_count = cluster_count.rename(columns={'Transient_Type': 'Transient_Type_Count'})

    #
    # Create markers for each transient types and store it in a dictionary
    #
    ttypes_markers = ["*","X","o","^","P","D","s"]
    for i, val in enumerate(zip(data.Transient_Type.unique(), ttypes_markers)):
        m_ttypes_dict[val[0]] = val[1]
    #
    # Create a dataframe for transient type
    #
    ttype_df = pd.DataFrame.from_dict(m_ttypes_dict, orient='index')
    ttype_df = ttype_df.reset_index()
    ttype_df.columns = ["Transient_Type", "Transient_Type_Marker"]
    #
    # Generate colors for each cluster
    #
    n_clusters = len(data.Cluster_ID.unique())
    c_clusters = mcp.gen_color(cmap="Set2", n=n_clusters)
    #
    # Create a dict() of colors for each clusters and sort the cluster id
    #
    for i, val in enumerate(zip(data.Cluster_ID.unique(), c_clusters)):
        c_clusters_dict[val[0]] = val[1]
    #
    c_clusters_dict = OrderedDict(sorted(c_clusters_dict.items(), key=lambda t: t[0]))
    #
    # Create the legends for transient types and sub-types
    #
    legend_elements_1 = [Line2D([], [], marker=f"{v[1]}", color=f"lightgray",ls='',
                                markeredgecolor='black', markeredgewidth=1.25, label=f"{v[0]}", markersize=8)
                         for i, v in enumerate(zip(ttype_df["Transient_Type"], ttype_df["Transient_Type_Marker"]))]
    legend_elements_2 = [Line2D([0], [0], marker="s", color=f"{c_clusters_dict[v]}",ls='', label=f"Cluster {v+1}: "
                         f"{((cluster_count.loc[cluster_count['Cluster_ID']==v,'Transient_Type_Count'].reset_index()['Transient_Type_Count'][0])/data.shape[0])*100:.1f}%",
                         markersize=8) for i, v in enumerate(c_clusters_dict.keys())]
    #
    # Plot the figure
    #
    plt.figure(figsize=(18, 8))
    plt.title( f"Clustering: {label2} ---- Feature Extraction: {label1}", **csfont, size=16)
    plt.xlabel(f'{label3} Features #1',size=12, labelpad=8, **csfont)
    plt.ylabel(f'{label3} Features #2',size=12, labelpad=8, **csfont)
    #
    # Plot the convex-hull
    #
    if convex_hull:
        #
        plot_kwds = {'alpha': 0.90, 's': 80, 'linewidths': 1.50}
        #
        for i in data.Cluster_ID.unique():
            #
            # Plot the convex-hull
            #
            points = data[data.Cluster_ID == i][['Feature#1', 'Feature#2']].values
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            #
            # Get the hex-code for the cluster id and fill the convex-hull
            #
            data_ = data.loc[data['Cluster_ID'] == i]
            plt.fill(x_hull, y_hull, alpha=0.130, c=data_["Cluster_Color_ID"].unique()[0], lw=2)
    else:
        #
        plot_kwds = {'alpha': 0.90, 's': 120, 'linewidths': 1.50}
    #
    # Group the dataframe by transient-type marker
    #
    for marker, d in data.groupby('Transient_Type_Marker'):
        #
        # Plot the data
        #
        plt.scatter(x=d["Feature#1"], y=d["Feature#2"], c=d["Cluster_Color_ID"], marker=marker,
                    edgecolor='white', **plot_kwds)
    #
    # Plot the grid
    #
    plt.grid(color='lightgray', linestyle='dotted', zorder=0)
    #
    # Plot the legends
    #
    legend_1 = plt.legend(handles=legend_elements_2, loc='upper right', bbox_to_anchor=(1.13, 0.4),
                          title= "Cluster Information",
                          title_fontsize=12, fancybox=True, fontsize=11)
    legend_2 = plt.legend(handles=legend_elements_1, loc='lower right', title= "Transient Types",
                          title_fontsize=12, bbox_to_anchor=(1.11, 0.5), fancybox=True, fontsize=11)
    #
    #
    #
    plt.gca().add_artist(legend_1)
    plt.tight_layout(pad=2.0)
    plt.savefig(f'images/{lc_type}/clustering/{method}_{extract_type}_{viz_method}_clusters.png', bbox_inches='tight')
    print(f"\nClusters are generated in - images/{lc_type}/clustering/ - folder!\n")
    return


def visualize_sub_clusters(data=None, viz_method="umap", method=None, extract_type=None, lc_type='transients'):
    """"
    Generates a plot to visualize the sub-clusters

    Parameters
    ----------
    data: Pandas DataFrame
        dataframe with the distribution information

    extract_type: String
        feature extraction methods - k_pca, tsfresh, or vae

    method: String
        clustering methods - birch or hdbscan

    viz_method: String
        visualization methods - umap or tsne

    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'


    Returns
    -------
    Generates images in -- images/{lc_type}/clustering/ - folder

    """

    #
    # Declare all the variables
    #
    m_ttypes_dict = dict()
    plot_kwds = {'alpha': 0.90, 's': 200, 'linewidths': 1.80}
    bbox_to_anchor_param = {4: [(1.35, 1.8), (1.3, 0.5)], 3: [(1.155, 3.1), (1.135, 0.6)],
                            2: [(1.155, 2.1), (1.135, 0.6)]}
    #
    # Create a 'images/{lc_type}/clustering' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/clustering/'):
        os.makedirs(f'images/{lc_type}/clustering/')
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
    try:
        if extract_type not in ["k_pca", "tsfresh", "vae"]:
            raise TypeError(f"\nTypeError: '{extract_type}' is not a valid extract_type!"
                            f"\nPlease provide the extract_type as - 'k_pca' , 'tsfresh', or 'vae'")
        else:
            #
            # Generate the labels
            #
            if extract_type == "k_pca":
                label1 = "Kernel PCA"
            elif extract_type == "tsfresh":
                label1 = "TSFresh"
            elif extract_type == "vae":
                label1 = "Variational Auto-Encoder"
    except Exception as e:
        print(e)
        exit()
    try:
        if method not in ["birch", "hdbscan"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid method!"
                            f"\nPlease provide the method as - 'birch' or 'hdbscan'")
        else:
            #
            # Generate the labels
            #
            if method == "birch":
                label2 = "Birch"
            elif method == "hdbscan":
                label2 = "HDBSCAN"
    except Exception as e:
        print(e)
        exit()
    try:
        if viz_method not in ["umap", "tsne"]:
            raise TypeError(f"\nTypeError: '{viz_method}' is not a valid viz_method!"
                            f"\nPlease provide the viz_method as - 'umap' or 'tsne'")
        else:
            #
            # Generate the labels
            #
            if viz_method == "umap":
                label3 = "UMAP"
            elif viz_method == "tsne":
                label3 = "TSNE"
    except Exception as e:
        print(e)
        exit()
    #
    # Calculate the total number of clusters
    # Plot the clusters based on odd/even counts
    #
    n_clusters = len(data.Cluster_ID.unique())
    clusters = sorted(data.Cluster_ID.unique())
    #
    q, r = divmod(n_clusters, 2)
    if (n_clusters <= 2 and r == 0) or (r != 0):
        n_row, n_col = n_clusters, 1
    else:
        n_row, n_col = q, 2
    #
    # Create markers for each transient types and store it in a dictionary
    #
    ttypes_markers = ["*", "X", "o", "^", "P", "D", "s"]
    #
    for i, val in enumerate(zip(data.Transient_Type.unique(), ttypes_markers)):
        m_ttypes_dict[val[0]] = val[1]
    #
    # Create a dataframe for transient sub-type info
    #
    tstype_df = data[["Transient_Sub_Type", "Transient_Sub_Type_Color", "Transient_Type_Marker"]]
    tstype_df = tstype_df.drop_duplicates(subset='Transient_Sub_Type', keep="first")
    #
    # Create a dataframe for transient type info
    #
    ttype_df = data[["Transient_Type", "Transient_Type_Marker"]]
    ttype_df = ttype_df.drop_duplicates(subset='Transient_Type', keep="first")
    #
    # Create the legends for transient types and sub-types
    #
    legend_elements_1 = [Line2D([], [], marker=f"{v[1]}", color=f"{v[2]}", label=f"{v[0]}", ls='', markersize=8)
                         for i, v in enumerate(zip(tstype_df["Transient_Sub_Type"],
                            tstype_df["Transient_Type_Marker"],
                            tstype_df["Transient_Sub_Type_Color"]))]

    legend_elements_2 = [Line2D([], [], marker=f"{v[1]}", color=f"lightgray", label=f"{v[0]}", ls='',
                         markeredgecolor='black', markeredgewidth=1.25, markersize=8)
                         for i, v in enumerate(zip(ttype_df["Transient_Type"],
                            ttype_df["Transient_Type_Marker"]))]
    #
    # Plot the figures
    #
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(15, 15))
    #
    #
    #
    plt.suptitle(f"Clustering: {label2} --- Feature Extraction: {label1}", **csfont, fontsize=22)
    #
    #
    #
    c_id = 0
    for r in range(n_row):
        if n_clusters % 2 == 0 and n_clusters > 2:
            for c in range(n_col):
                #
                #
                #
                axs[r, c].set_title(f"Cluster : {clusters[c_id]+1}", fontsize=18, **csfont)
                axs[r, c].set_xlabel(f'{label3} Feature #1', fontsize=13, **csfont)
                axs[r, c].set_ylabel(f'{label3} Feature #2', fontsize=13, **csfont)
                axs[r, c].tick_params(axis='x', labelsize=10)
                axs[r, c].tick_params(axis='y', labelsize=10)
                #
                # Plot the convex-hull
                #
                points = data[data.Cluster_ID == clusters[c_id]][['Feature#1', 'Feature#2']].values
                hull = ConvexHull(points)
                x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
                y_hull = np.append(points[hull.vertices, 1],points[hull.vertices, 1][0])
                #
                # Get the hex-code for the cluster id and fill the convex-hull
                #
                df1 = data.loc[data['Cluster_ID'] == clusters[c_id]]
                axs[r, c].fill(x_hull, y_hull, alpha=0.135, c=df1["Cluster_Color_ID"].unique()[0], lw=4)
                #
                # Group the dataframe by transient-type marker
                #
                for marker, df2 in df1.groupby('Transient_Type_Marker'):
                    #
                    # Group the dataframe by transient sub-type
                    #
                    for tstype, df3 in df2.groupby('Transient_Sub_Type'):
                        #
                        # Plot the data
                        #
                        axs[r, c].scatter(x=df3["Feature#1"], y=df3["Feature#2"], label=tstype,
                                          c=df3["Transient_Sub_Type_Color"].unique()[0],
                                          marker=marker, edgecolor='white', **plot_kwds)
                #
                # Plot the grid
                #
                axs[r, c].grid(color='lightgray', linestyle='dotted', zorder=0)
                #
                c_id += 1

        else:
            #
            #
            #
            axs[r].set_title(f"Cluster : {clusters[c_id]+1}", fontsize=18, **csfont)
            axs[r].set_xlabel(f'{viz_method} Feature #1', fontsize=13, **csfont)
            axs[r].set_ylabel(f'{viz_method} Feature #2', fontsize=13, **csfont)
            axs[r].tick_params(axis='x', labelsize=10)
            axs[r].tick_params(axis='y', labelsize=10)
            #
            # Plot the convex-hull
            #
            points = data[data.Cluster_ID == clusters[c_id]][['Feature#1', 'Feature#2']].values
            hull = ConvexHull(points)
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
            #
            # Get the hex-code for the cluster id and fill the convex-hull
            #
            df1 = data.loc[data['Cluster_ID'] == clusters[c_id]]
            axs[r].fill(x_hull, y_hull, alpha=0.135, c=df1["Cluster_Color_ID"].unique()[0], lw=4)
            #
            # Group the dataframe by transient-type marker
            #
            for marker, df2 in df1.groupby('Transient_Type_Marker'):
                #
                # Group the dataframe by transient sub-type
                #
                for tstype, df3 in df2.groupby('Transient_Sub_Type'):
                    #
                    # Plot the data
                    #
                    axs[r].scatter(x=df3["Feature#1"], y=df3["Feature#2"], label=tstype,
                                   c=df3["Transient_Sub_Type_Color"].unique()[0],
                                   marker=marker, edgecolor='white', **plot_kwds)
            #
            # Plot the grid
            #
            axs[r].grid(color='lightgray', linestyle='dotted', zorder=0)
            #
            c_id += 1
    #
    # Plot the legends
    #
    if n_clusters in bbox_to_anchor_param.keys():
        legend_1 = plt.legend(handles=legend_elements_1, loc='upper right', title= "Transient Sub-Types", ncol=1,
                              title_fontsize=12, bbox_to_anchor=bbox_to_anchor_param[n_clusters][0],
                              fancybox=True, fontsize=11)

        legend_2 = plt.legend(handles=legend_elements_2, loc='lower right', title= "Transient Types",
                              title_fontsize=12, bbox_to_anchor=bbox_to_anchor_param[n_clusters][1],
                              fancybox=True, fontsize=11)

    else:
        legend_1 = plt.legend(handles=legend_elements_1, loc='upper right', title= "Transient Sub-Types", ncol=1,
                              title_fontsize=12, bbox_to_anchor=(1.0, 0.0), fancybox=True, fontsize=11)

        legend_2 = plt.legend(handles=legend_elements_2, loc='lower right', title= "Transient Types",
                              title_fontsize=12, bbox_to_anchor=(1.0, 0.0), fancybox=True, fontsize=11)
    #
    #
    #
    plt.gca().add_artist(legend_1)
    fig.tight_layout(pad=2.0)
    fig.savefig(f'images/{lc_type}/clustering/{method}_{extract_type}_{viz_method}_sub-cluster.png', bbox_inches='tight')
    print(f"\nSub-Clusters are generated in - images/{lc_type}/clustering/ - folder!\n")
    return


if __name__ == '__main__':

    data = load_latent_space(extract_type='vae')
    X_train, labels = data['data'], data['labels']
    data_info = generate_data(lc_type="transients", filename="birch_vae.pickle", method='birch',
                              viz_method='tsne', image_path="processed/", X=X_train, lc_labels=labels)
    visualize_clusters(data=data_info, convex_hull=True, viz_method="umap", method="birch",
                       extract_type="vae")
    visualize_sub_clusters(data=data_info, viz_method="umap", method="birch",
                           extract_type="vae")

