#
# Import all the dependencies
#
import os
import re
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



class TESS_Transients(object):

    """
    TESS_Transients is a pre-processing tool for generating transients from NASA's TESS telescope.

    This class generates processed transients for the IAU Name in the "transients.txt" file.
    The object names are available at - "https://www.wis-tns.org"

    Parameters
    ----------

    path: string
        Stores the absolute path of the current file

    transients: dict
        Stores the IAU Name and object type as a key-value pair from the .txt file - transients.txt

    transients_info: pandas-dataframe
        Stores all the meta-data for the available IAU Name in "transients.txt".
        It has 13 columns in the following order -

    ----------------------------------------------------------------------
    -------------------------------- Columns -----------------------------
    ----------------------------------------------------------------------
    ['sector','ra','dec','magnitude at discovery','time of discovery in TESS JD',
    'SN','classification', 'IAU name','discovery survey','cam','ccd', 'column','row']

    """


    def __init__(self):

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.transients = None
        self.transients_info = None


    def generate_transients(self, filename=None):

        """
        Generates a dictionary for the available transients and their label (as to transient_labels. pickle)
        and a dataframe for all transients' metadata (as to transients_info.pickle)

        Parameters
        ----------
        filename: .txt file
            It contains metadata for all the available transients

        Returns
        -------
        duplicate_transient_ids: csv file
            Stores all the duplicate IAU Name in the current folder
        transient_labels: pickle file
            Stores all the transients and their label as a dict
        transients_info: pickle file
            Stores all the transients metadata
        """

        #
        # Verify the type of filename
        # It should be a .txt file
        # TO DO --- .csv file
        #
        try:
            f_type = re.search('\.[ctTC]*?\w+', filename)
            if f_type:
                f_type=f_type.group(0)
                f_type = f_type.lower()
                if f_type == ".csv":
                    raise NotImplementedError(f"NotImplementedError: Expected a '.txt' file but got '{f_type}' file")
                if f_type != ".txt" and f_type != ".csv":
                    raise TypeError(f"TypeError: Expected a '.txt' file but got '{f_type}' file")

        except Exception as e:
            print(e)
            return

        #
        # Create a '/data' folder if it doesn't exists already
        #
        if not os.path.exists('data'):
            os.makedirs('data')

        #
        # Read transients meta-data from the .txt file and store it as a pandas dataframe
        #
        if f_type == ".txt":
            data = pd.read_csv(filename, sep=r'\s+', header=None)

        col_ = data.shape[1]
        #
        # Verify the number of columns in the dataframe
        # The text file should have 13 columns
        #
        try:
            if col_!=13:
                raise ValueError(f"ValueError: Expected 13 columns in the '.txt' file, but got {col_} column(s)."
                                 f"\nMake sure to have columns according to the following sequence --\n"
                                 f"['sector','ra','dec','magnitude at discovery','time of discovery in TESS JD',\n"
                                 f"'SN','classification', 'IAU name','discovery survey','cam','ccd',"
                                 f"'column','row']")
        except Exception as e:
            print(e)
            return

        #
        # Rename the column of the pandas dataframe
        #
        column_names = ["sector","ra","dec","magnitude at discovery","time of discovery in TESS JD",
                        "SN","classification", "IAU name","discovery survey","cam","ccd","column","row"]

        data.columns = column_names

        #
        # Check for duplicate IAU Name
        #
        duplicate_data = data[data.duplicated(subset=['IAU name'])]
        duplicated_id = duplicate_data['IAU name'].to_list()

        #
        # If any duplicate IAU Name exists then drop it from the dataframe
        #
        if duplicated_id:

            data = data.drop_duplicates(subset=['IAU name'], keep='last')
            data = data.reset_index(drop=True)
            self.transients_info = data

        #
        # Store the transients label
        #
        transients_label = dict(zip(data['IAU name'], data['classification']))
        self.transients = transients_label

        #
        # Store transients label, transients metadata, and duplicate IAU Name
        #
        with open('data/transient_labels.pickle', 'wb') as file:
            pickle.dump(self.transients, file)

        with open('data/transients_info.pickle', 'wb') as file:
            pickle.dump(self.transients_info, file)

        with open(self.path + f"/duplicate_transient_ids.csv", 'w', newline="") as f:

            write = csv.writer(f)
            for i in duplicated_id:
                write.writerow([i])
        f.close()

        print(f"\nFiles are generated in the /data folder!"
              f"\nDuplicate transient ids are stored in the current folder.\n")



    def read_transient_labels_from_pickle(self):

        """
        Returns
        -------
        transient_labels: dict
            Load all the transient labels from the pickle file
        """

        with open('data/transient_labels.pickle', 'rb') as file:
            transient_labels = pickle.load(file)

        return transient_labels



    def read_transients_info_from_pickle(self):

        """
        Returns
        -------
        transients_info: pandas-dataframe
            Load the transients metadata from the pickle file
        """

        with open('data/transients_info.pickle', 'rb') as file:
            transients_info = pickle.load(file)

        return transients_info



    def binned_transients(self, df=None, interval = "0.5D", time_col = "relative_time", uncert="e_cts"):
        """
        Generate binned transients

        Parameters
        ----------
        df: pandas-dataframe
            dataframe with columns relative_time, cts and e_cts
        interval: string (default = "0.5D")
            scalar + unit, eg. 0.5D, Units: Day: D, Minute: T, Second: S
        time_col: string (default = "relative_time")
            time column name
        uncert: string (default = "e_cts")
            uncertainty column name


        Returns
        -------
        binned_data: pandas-dataframe
            binned dataframe

        """

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


        return binned_data


    def save_transients(self, path=None):

        """
        Generate pre-processed transients

        Parameters
        ----------
        path: path
            folder path, which contains all the unprocessed transients
            in the format -- lc_{IAU Name}_processed.csv

        Notes
        -----
        All the unprocessed transients in the format -- lc_{IAU Name}_processed.csv
        contains columns - relative_time, cts and e_cts

        The transients generated has 60 data samples

        Returns
        -------
        Pre-processed transients as - {IAU Name}.csv.gz in /processed_transients

        unsearched_transients: pickle file
            It stores all the files in the format -- lc_{IAU Name}_processed.csv
            where data cannot be stored

        """

        transient_ids = list()
        cleaned_files = list()
        unsearched_transients = list()

        #
        # Create '/processed_transients' folder if it does not exists already
        #
        if not os.path.exists('processed_transients'):
            os.makedirs('processed_transients')

        #
        # Search for .csv file in the format --- lc_{IAU Name}_processed.csv ---
        # Store the files in -- cleaned_files
        # Store the IAU Names in -- transient_ids
        #
        for file in os.listdir(path):
            if bool(re.search(r"\blc_\d+[a-zA-Z]{1,5}_processed\b", file)):
                cleaned_files.append(file)
                transient_ids.extend(re.findall("_(.*?)_processed", file))


        for i, file in enumerate(cleaned_files):

            try:
                #
                # Load the .csv to pandas dataframe
                #
                data = pd.read_csv(path+f"{file}")
                #
                # Calculate the binned data with default -- interval = "0.5D" and time_col = "relative_time"
                #
                binned_data = self.binned_transients(data)
                #
                # Drop the index and rename the columns
                #
                binned_data = binned_data.reset_index(drop=True)
                binned_data = binned_data.rename({'relative_time':'time', 'cts':'flux', 'e_cts':'flux_err'}, axis=1)
                #
                # Calculate the relative time with respect to the start time
                #
                binned_data['time'] = binned_data['time'] - binned_data['time'][0]
                #
                # Save the transients in '/processed_transients'
                #
                transient_path = self.path + f'/processed_transients/{transient_ids[i]}.csv.gz'
                binned_data.to_csv(path_or_buf=transient_path, header=True, index=False, compression='gzip')

            except Exception as e:
                unsearched_transients.append(file)
                print(f"Exception Raised!!\n--- {e}\n\n"f"{file} --- IAU Name: {transient_ids[i]} couldn't be stored.\n\n")
                continue

        #
        # Store all the available IAU Name in -- transient_ids.csv
        #
        if transient_ids:
            with open(self.path + f"/transient_ids.csv", 'w', newline="") as f:

                write = csv.writer(f)
                for i in transient_ids:
                    write.writerow([i])
            f.close()

        #
        # Store all the unavailable transients in -- unsearched_transients.pickle
        #
        if unsearched_transients:
            with open(self.path+f"/unsearched_transients.pickle", 'wb') as file:
                pickle.dump(unsearched_transients, file)

        print(f"\nTransients are generated and stored in /processed_transients.\n")



    def preprocess_transients(self, path=None):

        """
        Parameters
        ----------
        path: path
            folder path, which contains all the processed transients
            in the format -- {IAU Name}.csv.gz

        Returns
        -------
        transients: dict
            Contains 'time' (key) values as a list of length 300 and
            'flux' (key) and 'flux_err' (key) as ndarrays of shape (n, 300)
            where 'n' is the total number transients generated.
        """

        file_list = list()
        file_exception = list()
        flux = list()
        meta_data = list()
        flux_err = list()
        transients = dict()


        scaler = MinMaxScaler()

        #
        # Store all the available file from /processed_transients
        #
        for file in os.listdir(path):
            file_list.append(file)


        for file in file_list:
            try:
                data = pd.read_csv(path+f"{file}", compression='gzip')
                #
                # Store the transients where the shape is (60,1),
                # Transform the flux and flux_err using MinMaxScaler(), and
                # Replace the NaNs (if exists) with the mean value of the flux and the flux_err columns
                #
                #
                # Copy the flux and flux_err columns
                #
                data_filt = data[["flux","flux_err"]]
                #
                # Apply MinMaxScalar (feature range = [0,1])
                #
                data_ = scaler.fit_transform(data_filt)
                #
                # Replace the flux and flux_err values with the normalized values
                #
                data.flux = data_[:, 0]
                data.flux_err = data_[:, 1]
                flux_mean = np.nanmean(data_[:, 0])
                flux_err_mean = np.nanmean(data_[:, 1])
                #
                # Calculate the total NaNs
                #
                nan_count = data.flux.isna().sum()
                #
                if nan_count <= int(0.3 * len(data.flux)):
                    data_processed = data.copy(deep=True)
                    data_processed.flux.fillna(value=flux_mean, inplace=True)
                    data_processed.flux_err.fillna(value=flux_err_mean, inplace=True)
                    #
                    # Reshape the dataframe to (60,3)
                    #
                    if (len(data_processed.flux) != 60):
                        n_row = data_processed.shape[0]
                        diff = 60 - n_row
                        arr = np.zeros((diff, 3))
                        data_processed = data_processed.append(pd.DataFrame(arr, columns=data.columns), ignore_index=True)
                    #
                    #
                    #
                    flux_err.append(data_processed.flux_err.values.tolist())
                    flux.append(data_processed.flux.values.tolist())
                    meta_data.append(file[:-7])


            except Exception as e:
                print(f"\nException Raised: {e}\n"f"Check : {file[:-7]}, {file}")
                #
                # Store the transients for which the flux couldn't be stored
                #
                file_exception.append(file)
                continue

        #
        # Store the time, flux and flux_err values in the dictionary -- transients
        #
        flux = np.array(flux)
        flux_err = np.array(flux_err)
        transients['flux'] = flux
        transients['flux_err'] = flux_err
        transients['time'] = np.arange(0, 30, 0.5)
        transients['metadata'] = meta_data

        #
        # Store the files
        #
        with open(f"data/transients.pickle", 'wb') as file:
            pickle.dump(transients, file)

        #
        # Store all the unavailable transients in -- file_exception.pickle
        #
        if file_exception:
            with open(self.path+f"/file_exception.pickle", 'wb') as file:
                pickle.dump(file_exception, file)
            print("\n Flux values for some IAU Name couldn't be stored.\n"
                  "Please refer file_exception.pickle file!\n")

        print(f"\nTransients are pre-processed and stored in /data.\n")




    def generate_transient_image(self, path=None):

        """
         Generates images for all the available transients in '/images'

        Parameters
        ----------
        path: path
            folder path, which contains all the processed transients
            in the format -- {IAU Name}.csv.gz
        """
        #
        file_list = list()
        csfont = {'fontname':'Comic Sans MS'}
        #
        # Create a '/images' folder if it does not exists already
        #
        if not os.path.exists('images'):
            os.makedirs('images')
        #
        # Load the transient_labels.pickle file
        #
        with open('data/transient_labels.pickle', 'rb') as file:
            ids = pickle.load(file)
        #
        # Store all the available file from /processed_transients
        #
        for file in os.listdir(path):
            file_list.append(file)
        #
        # Generate the images
        #
        n_row, id = 3, 0
        q, r = divmod(len(file_list), n_row)
        if r != 0: q += 1
        batch_size = q

        for batch in range(batch_size):
            fig = plt.figure(figsize=(1024, 2048))
            fig, axs = plt.subplots(nrows=n_row, ncols=1, figsize=(18,15))

            k = id
            for i in range(n_row):

                try:
                    if id < len(file_list):
                        t_id = file_list[id][:-7]
                        label = ids[t_id]
                        axs[i].set_title(f"IAU Name : {t_id}  -----  Label : {label}", fontsize=25, **csfont)
                        binned_lc = pd.read_csv(f"{path}/{file_list[id]}", compression='gzip')
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
            fig.savefig(self.path+f"/images/image_{k}_{id-1}.png", bbox_inches ="tight", orientation ='landscape')

        print(f"\nImages are available in /images.\n")






if __name__ == '__main__':


    create_transients = TESS_Transients()

    create_transients.generate_transients(filename="transients.txt")
    create_transients.save_transients(path="./processed_curves/")
    create_transients.preprocess_transients(path="./processed_transients/")
    create_transients.generate_transient_image(path='./processed_transients')