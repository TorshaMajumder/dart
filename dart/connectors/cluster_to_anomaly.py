#
# Import all the dependencies
#
import os
import pickle
import random
import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
from dart.feature_extraction.UMAP_ import UMap
from dart.anomaly_detection.OneclassSVM import OneClassSVM_


def create_clusters(method=None, extract_type=None, anomaly_method=None):
    #
    #
    #
    try:
        if extract_type not in ["k_pca", "tsfresh", "vae", "isomap", "umap"]:
            raise TypeError(f"\nTypeError: '{extract_type}' is not a valid type!"
                            f"\nPlease provide the type as - 'k_pca' , 'tsfresh', 'isomap', 'umap', or 'vae'")
    except Exception as e:
        print(e)
        exit()

    try:
        if method not in ["hdbscan", "birch"]:
            raise TypeError(f"\nTypeError: '{method}' is not a valid type!"
                            f"\nPlease provide the type as - 'hdbscan' or 'birch'.")
    except Exception as e:
        print(e)
        exit()

    try:
        if anomaly_method not in ["svm", "lof"]:
            raise TypeError(f"\nTypeError: '{anomaly_method}' is not a valid type!"
                            f"\nPlease provide the type as - 'svm' or 'lof'.")
    except Exception as e:
        print(e)
        exit()
    #
    # Load the data
    #
    try:
        with open(f"../latent_space_data/transients/{extract_type}.pickle", 'rb') as f:
            latent_space_data = pickle.load(f)
    except Exception as e:
        print(f"\n\nException Raised: {e}\n\n")

    try:
        with open(f"../transients/plasticc_labels_.pickle", 'rb') as f:
            labels = pickle.load(f)
    except Exception as e:
        print(f"\n\nException Raised: {e}\n\n")

    try:
        with open(f"../results/clustering/transients/{method}_{extract_type}.pickle", 'rb') as f:
            cluster_dict = pickle.load(f)
    except Exception as e:
        print(f"\n\nException Raised: {e}\n\n")
    #
    # Prepare the data for cluster to use them in bokeh plots
    #
    for c in np.unique(cluster_dict["clusters"]):
        #
        try:
            idx = np.where((cluster_dict["clusters"] == c))
            cluster_data = latent_space_data["data"][idx]
            cluster_ids = np.array(latent_space_data["labels"])[idx]
            cluster_labels = np.array([labels[k] for k in cluster_ids])
            #
            umap_ = UMap(lc_type="transients")
            umap_data = umap_.fit_transform(X_train=cluster_data, visualize=True)
            #
            if anomaly_method == "svm":
                svm = OneClassSVM_()
                anomaly_score = svm.fit_predict(cluster_data)
            elif anomaly_method == "lof":
                pass
            #
            cluster_labels = cluster_labels.reshape((len(cluster_labels), 1))
            anomaly_score = anomaly_score.reshape((len(anomaly_score), 1))
            #
            # Create a dataframe for class, score, and color
            #
            c_ttypes_dict = dict()
            n_ttypes = len(np.unique(cluster_labels))
            palette = sns.color_palette(cc.glasbey, as_cmap=True)
            c_ttypes = random.choices(palette, k=n_ttypes)
            for i, val in enumerate(zip(np.unique(cluster_labels), c_ttypes)):
                c_ttypes_dict[val[0]] = val[1]
            color_df = pd.DataFrame.from_dict(c_ttypes_dict, orient="index").reset_index()
            color_df.columns = ["class", "color"]
            cluster_info = np.concatenate((cluster_labels, anomaly_score), axis=1)
            df = pd.DataFrame(cluster_info, columns=["class", "score"])
            df = df.merge(color_df, on="class", how="inner")
            #
            # Store the cluster information
            #
            df_dict = {"cluster_data": cluster_data, "cluster_label": cluster_labels,
                       "cluster_ids": cluster_ids, "df": df,
                       "umap": umap_data, "index": idx}

            if not os.path.exists(f"../results/anomaly_detection/{anomaly_method}/{method}_{extract_type}"):
                os.makedirs(f"../results/anomaly_detection/{anomaly_method}/{method}_{extract_type}")
            #
            # Store the file in -- '/results/anomaly_detection/{anomaly_method}/{method}_{extract_type}/' folder
            #
            with open(f"../results/anomaly_detection/{anomaly_method}/{method}_{extract_type}/cluster_{c}.pickle", 'wb') as file:
                pickle.dump(df_dict, file)
        except Exception as e:
            print(f"\n\nException Raised: {e}\n\n")


if __name__ == '__main__':

    create_clusters(method="hdbscan", extract_type="umap", anomaly_method="svm")