#
# Import all the dependencies
#
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from TESS.feature_extraction.TSFresh import TSFresh
#
# Set global parameters
#
csfont = {'fontname': 'Comic Sans MS'}
#
# Create a '/images' folder if it does not exists already
#
if not os.path.exists('images'):
    os.makedirs('images')


def get_feature_importance(lc_type='transients', imp_method=None):
    """
    Generates a histogram to visualize the feature importance for TSFresh data

    Parameters
    ----------
    lc_type: String (default - 'transients')
        type of light curve - 'transients' or 'transits'

    imp_method: String
        feature importance method type - k_pca (Kernel_PCA) or urf (Unsupervised Random Forest)

    Returns
    --------
    Generates images in -- images/{lc_type}/feature_importance/ - folder

    """
    #
    # Declare the variables
    #
    sorted_id, new_sorted_id, sorted_imp = list(), list(), list()
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
    #
    #
    try:
        with open(f"../latent_space_data/{lc_type}/tsfresh.pickle", 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError as e:
        print(f"\nFileNotFoundError: Data cannot be loaded!"
              f"\nPlease verify if the folder - ../latent_space_data/{lc_type}/tsfresh.pickle - exists.\n")
        exit()

    try:
        if imp_method not in ["k_pca", "urf"]:
            raise TypeError(f"\nTypeError: '{imp_method}' is not a valid extract_type!"
                            f"\nPlease provide the extract_type as - 'k_pca' or 'urf'")
        else:
            #
            # Generate the labels
            #
            if imp_method == "k_pca":
                label1 = "Kernel PCA"
            elif imp_method == "urf":
                label1 = "Unsupervised Random Forest"
    except Exception as e:
        print(e)
        exit()
    #
    # Extract the feature importance dictionary
    #
    feature_imp = data["feature_imp"]
    #
    # Extract the sorted feature and feature importance lists
    #
    for k in feature_imp.keys():
        sorted_id.append(k)
        sorted_imp.append(feature_imp[k])
    #
    # Generate x-ticks
    #
    features = np.arange(len(sorted_id))
    #
    # Normalize the feature importance
    #
    norm_sorted_imp = sorted_imp/np.max(sorted_imp)
    #
    # Add a new line char when the length of feature name > 25
    #
    for i in range(len(sorted_id)):
        if len(sorted_id[i]) > 25:
            s = sorted_id[i][0:20]+"-\n"+sorted_id[i][20:40]+"-\n"+sorted_id[i][40:]+"\n"
            new_sorted_id.append(s)
        else:
            new_sorted_id.append(sorted_id[i])
    #
    # Create a 'images/{lc_type}/feature_importance' folder if it does not exists already
    #
    if not os.path.exists(f'images/{lc_type}/feature_importance/'):
        os.makedirs(f'images/{lc_type}/feature_importance/')
    #
    # Generate the plot
    #
    plt.figure(figsize=(15, 8))
    plt.title(f"Feature Importance: TSFresh and {label1}", **csfont, size=12)
    plt.grid(color='lightgray', linestyle='dashed',linewidth=1.0, zorder=0)
    plt.bar(features, norm_sorted_imp, width=0.4, align='center',
            edgecolor='black', color='lightsteelblue', zorder=3)

    plt.xlim(-1, 10)
    plt.ylim(0, 1.1)
    plt.margins(x=0, y=0)
    plt.xlabel('Features',size=12, labelpad=2, **csfont)
    plt.ylabel('Importance Score',size=12, labelpad=8, **csfont)
    plt.xticks(features, new_sorted_id, rotation=0, fontsize=6.5)
    plt.tight_layout(pad=1.5)
    plt.savefig(f'images/{lc_type}/feature_importance/tsfresh_{imp_method}.png', bbox_inches='tight')

    print(f"\nThe feature importance plot is generated and stored in - "
          f"images/{lc_type}/feature_importance/ - folder!\n")

    return


if __name__ == '__main__':

    tsfresh = TSFresh(lc_type="transients")
    tsfresh.get_important_features(path="../transients/data/tsfresh_data.pickle", method="urf")
    get_feature_importance(lc_type="transients", imp_method="urf")