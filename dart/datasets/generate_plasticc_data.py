import os
import pickle
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, griddata
#
# Create '/transients' folder if it does not exist already
#
if not os.path.exists("../transients"):
    os.makedirs("../transients")
#
# Create '/transients/plasticc_data' folder if it does not exist already
#
if not os.path.exists("../transients/plasticc_data"):
    os.makedirs("../transients/plasticc_data")
#
# Set all the parameters
#
random.seed(0)
path_to_store = "../transients/plasticc_data"


def binned_transients(df=None, interval="1D", time_col="relative_time", uncert="r_uncert"):

    binned_data = df.copy()
    #
    # square e_cts to get variances
    #
    binned_data[uncert] = np.power(binned_data[uncert], 2)
    #
    # bin and find avg variance and avg mean per bin
    #
    binned_data.index = pd.TimedeltaIndex(df[time_col], unit="D").round(interval)
    binned_data = binned_data.resample(interval, origin="start").mean()
    binned_data[time_col] = binned_data.index / pd.to_timedelta('1D')
    binned_data.index = binned_data.index / pd.to_timedelta(interval)
    #
    # sqrt avg vars to get uncertainty in stds
    #
    binned_data[uncert] = np.power(binned_data[uncert], 0.5)
    binned_data = binned_data.reset_index(drop=True)
    binned_data = binned_data.set_index("relative_time", drop=False)

    return binned_data


def interpolate(x, y, x_new, id):

    index = np.arange(len(y))
    interp = np.array(y)
    
    try:
        interp[np.isnan(interp)] = griddata(index[~np.isnan(y)], y[~np.isnan(y)],
                                            index[np.isnan(y)], method="nearest")
    except Exception as e:
        print(f"\nException Raised: {e},\n{id}")
    try:
        func = interp1d(x, interp, fill_value="extrapolate", kind="linear")
        y_new = func(x_new)
        return y_new
    except Exception as e:
        print(f"\nException Raised: {e}\n\n{interp},\n{x}, \n{id}")


def create_dataframe(df, redshift, mwebv, id, type_, binning=False, bin_interval="1D",
                     l_bound=-30.0, u_bound=70.0):

    #
    # Store the unique filters
    #
    bands = df["passband"].unique()
    bands.sort()
    #
    # Define the range and interval
    #
    curve_range, interval = (l_bound, u_bound), 3
    #
    # Create the feature list for the new dataframe
    #
    feature_list = ["relative_time"]
    for band in bands:
        flux = f"{band}_flux"
        uncert = f"{band}_uncert"
        feature_list.append(flux)
        feature_list.append(uncert)
    #
    # Initialize the dataframe with zeros
    #
    new_df = pd.DataFrame(0.0, index=np.arange(len(df)), columns=feature_list)
    new_df["relative_time"] = df["time"]
    new_df = new_df.set_index("relative_time")
    #
    # Store the values for each filter per row
    #
    for i, t in enumerate(df["time"]):

        row = df[df["time"] == t]
        band, flux, uncert = row.at[i,"passband"], row.at[i,"flux"], row.at[i,"fluxErr"]
        new_df.loc[t,f"{band}_flux"] = flux
        new_df.loc[t,f"{band}_uncert"] = uncert
    #
    # Select the data point within interval [l_bound, u_bound]
    #
    new_df = new_df.reset_index("relative_time")
    new_df = new_df.loc[(new_df["relative_time"] >= l_bound) & (new_df["relative_time"] <= u_bound)]
    #
    # Bin the data if binning is True
    #
    if binning:

        new_df = binned_transients(df=new_df, interval=bin_interval)
    #
    #
    #
    x_new = np.arange(curve_range[0], curve_range[1], interval)
    interpolate_df = pd.DataFrame()
    interpolate_df["relative_time"] = x_new
    #
    #
    #
    for band in bands:
        x = new_df.relative_time.values
        flux = new_df[f"{band}_flux"].values
        flux_new = interpolate(x, flux, x_new, id)
        interpolate_df[f"{band}_flux"] = flux_new
        uncert = new_df[f"{band}_uncert"].values
        uncert_new = interpolate(x, uncert, x_new, id)
        interpolate_df[f"{band}_uncert"] = uncert_new
    
    new_df = new_df.drop(columns="relative_time")
    new_df = new_df.reset_index()
    #
    # Add the metadata
    #
    interpolate_df["redshift"] = redshift
    interpolate_df["mwebv"] = mwebv
    interpolate_df["id"] = id+"-"+type_
    #
    # Store the dataframe to a CSV file
    #
    interpolate_df.to_csv(path_or_buf=f"{path_to_store}/{id}-{type_}.csv", index=False)


def generate_data(path=None, l_bound=-30.0, u_bound=70.0):
    """

    Parameters
    ----------
    path: String
        path to extract the data

    l_bound: int
        lower bound of the time step

    u_bound: int
        upper bound of the time step

    Returns
    -------
    "new_plasticc_objects.pickle" file under the current directory
    """
    #
    # Declare a dictionary to store the filtered object ids
    #
    lc_objects = dict()
    #
    # Read the files in the folder
    #
    filenames = os.listdir(path)
    #
    for i, filename in enumerate(filenames):
        #
        # Load each objects in the folder
        #
        file = open(f"{path}/{filenames[i]}", 'rb')
        data = pickle.load(file)
        object_id = list(data.keys())
        #
        # Create a list to store the filtered object ids for each file type
        #
        objects = list()
        #
        for j, id in enumerate(object_id):
            #
            try:
                #
                # Verify if there is any data within the range (l_bound, u_bound)
                #
                df = data[id].to_pandas()
                df = df.sort_values(by=["time"], ignore_index=True)
                new_df = df.loc[(df["time"] >= l_bound) & (df["time"] <= u_bound)]
                if len(new_df) != 0:
                    objects.append(id)
            except Exception as e:
                print(f"\nException Raised: {e}, \nObject ID: {id}")
                continue
        lc_objects[filenames[i]] = objects
    #
    with open(f"../transients/new_plasticc_objects.pickle", "wb") as file:
        pickle.dump(lc_objects, file)

def main():
    #
    # Declare the folder that contains the PLAsTiCC data set
    #
    plasticc_data = "saved_light_curves_oldandnewsims"
    l_bound = -30.0
    u_bound = 70.0
    #
    # Filter the objects that contains data within the timestamp (-30, 70)
    #
    generate_data(path=plasticc_data, l_bound=l_bound, u_bound=u_bound)
    #
    # Prepare the label list for normals and anomalies
    #
    label_list = ['AGN', 'CART', 'SNIa-91bg', 'SNIa', 'SNIa-x', 'SNIb', 'SNIc-BL', 'SNIc', 'SNII', 'SNIIb', 'SNIIn',
                  'ILOT', 'Kilonova', 'PISN', 'SLSN-I', 'TDE', 'uLens']
    normals = ['CART', 'SNIa-91bg', 'SNIa', 'SNIa-x', 'SNIb', 'SNIc-BL', 'SNIc', 'SNII', 'SNIIb', 'SNIIn']
    anomalies = ['Kilonova', 'uLens', 'ILOT', 'PISN', 'SLSN-I', 'TDE', 'AGN']
    #
    # Create a global dataframe to store the object ids and labels
    #
    label_df = pd.DataFrame(columns=["object_id", "label"])
    #
    # Declare if you want to have a sample to object ids and the number of objects in the samples
    #
    sample, n = True, 0
    #
    # Read the filtered objects in the - new_plasticc_objects.pickle
    #
    file = open(f"../transients/new_plasticc_objects.pickle", 'rb')
    filtered_object_ids = pickle.load(file)
    #
    # Read the files in the folder
    #
    filenames = os.listdir(plasticc_data)
    #
    # #samples for normals is set to 500
    # #samples for anomalies is set to 100
    # total samples - (500*10 + 100*7) = 5700
    #
    for i, filename in enumerate(filenames):
        #
        if label_list[i] in normals:
            n = 500
        elif label_list[i] in anomalies:
            n = 100
        else:
            continue
        #
        # Load each objects in the folder
        #
        file = open(f"{plasticc_data}/{filenames[i]}", 'rb')
        data = pickle.load(file)
        #
        # Load the object ids from the dictionary
        #
        object_id = filtered_object_ids[filenames[i]]
        #
        # Rename the label as - "object_id-"+"class"
        #
        rename_ids = list()
        #
        print(f"Transient Type: {label_list[i]}, n_samples: {n}\n")
        #
        # Verify if the #object_ids in a class > 1000
        #
        if sample:
            if len(object_id) > 1000:
                object_id = random.sample(object_id, n)
        #
        # Create a local dataframe to store the object ids and labels
        #
        label_file_df = pd.DataFrame(columns=["object_id", "label"])
        #
        # Rename the label as - "object_id-"+"class"
        #
        for o, obj in enumerate(object_id):
            rename_ids.append(obj + "-" + label_list[i])
        #
        # Store the object_ids and labels
        #
        label_file_df["object_id"] = rename_ids
        label_file_df["label"] = label_list[i]
        label_df = pd.concat([label_file_df, label_df])
        #
        # Pre-process the data for each object ids and
        # store it as a CSV in the folder - /plasticc_data
        #
        for j, id in enumerate(object_id):
            #
            try:
                #
                redshift = data[id].meta["redshift"]
                mwebv = data[id].meta["mwebv"]
                df = data[id].to_pandas()
                df = df.sort_values(by=["time"], ignore_index=True)
                create_dataframe(df=df, redshift=redshift, mwebv=mwebv, id=id,
                                 type_=label_list[i], binning=True, bin_interval="3D",
                                 l_bound=l_bound, u_bound=u_bound)

            except Exception as e:
                print(f"\nException Raised: {e}, \nObject ID: {id}")
                continue
    #
    # Store the object ids and labels as a dictionary
    #
    transient_labels = dict(zip(label_df["object_id"], label_df["label"]))
    #
    with open(f"../transients/plasticc_labels_.pickle", "wb") as file:
        pickle.dump(transient_labels, file)


if __name__ == '__main__':

    main()


    
