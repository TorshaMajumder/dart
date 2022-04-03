#
# Import all the dependencies
#
import os
import csv
import pickle
import numpy as np
import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler



class TESS_Lightcurves(object):
    """
    TESS_Lightcurves is a pre-processing tool for generating light curves from NASA's TESS telescope.

    This class generates light curves for the TOIs available at - "https://exofop.ipac.caltech.edu/tess/view_toi.php"

    It utilizes the Lightkurve package for pre-processing the light curves from authors - SPOC, QLP, and TESS-SPOC.
    It also stores the TOIs that do not have any data or could not download the target TIC ID using the Lightkurve package.


    Parameters
    ----------

    path: string
        Stores the absolute path of the current file

    toi: dict
        Stores the TOIs and TIC IDs as a key-value pair from the .csv file
        available at - "https://exofop.ipac.caltech.edu/tess/view_toi.php"

    toi_info: pandas-dataframe
        Stores all the relevant columns of the 'toi.csv' file
    ----------------------------------------------------------------------
    -------------------------------- Columns -----------------------------
    ----------------------------------------------------------------------
    ['TIC ID', 'TOI', 'RA', 'Dec', 'PM RA (mas/yr)', 'PM RA err (mas/yr)',
   'PM Dec (mas/yr)', 'PM Dec err (mas/yr)', 'Epoch (BJD) err','Period (days)',
   'Period (days) err', 'Duration (hours)','Duration (hours) err', 'Depth (mmag)',
   'Depth (mmag) err','Depth (ppm)', 'Depth (ppm) err', 'Planet Radius (R_Earth)',
   'Planet Radius (R_Earth) err', 'Planet Insolation (Earth Flux)','Planet Equil Temp (K)',
   'Planet SNR', 'Stellar Distance (pc)','Stellar Distance (pc) err', 'Stellar Eff Temp (K)',
   'Stellar Eff Temp (K) err', 'Stellar log(g) (cm/s^2)','Stellar log(g) (cm/s^2) err',
   'Stellar Radius (R_Sun)','Stellar Radius (R_Sun) err', 'Stellar Metallicity',
   'Stellar Metallicity err', 'Stellar Mass (M_Sun)','Stellar Mass (M_Sun) err', 'Epoch (BTJD)']


    References
    ----------
    Lightkurve: a Python package for Kepler and TESS data analysis

    @MISC{2018ascl.soft12013L,
    author = {{Lightkurve Collaboration} and {Cardoso}, J.~V.~d.~M. and
         {Hedges}, C. and {Gully-Santiago}, M. and {Saunders}, N. and
         {Cody}, A.~M. and {Barclay}, T. and {Hall}, O. and
         {Sagear}, S. and {Turtelboom}, E. and {Zhang}, J. and
         {Tzanidakis}, A. and {Mighell}, K. and {Coughlin}, J. and
         {Bell}, K. and {Berta-Thompson}, Z. and {Williams}, P. and
         {Dotson}, J. and {Barentsen}, G.},
    title = "{Lightkurve: Kepler and TESS time series analysis in Python}",
    keywords = {Software, NASA},
    howpublished = {Astrophysics Source Code Library},
    year = 2018,
    month = dec,
    archivePrefix = "ascl",
    eprint = {1812.013},
    adsurl = {http://adsabs.harvard.edu/abs/2018ascl.soft12013L},}

    """


    def __init__(self):

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.toi = None
        self.toi_info = None


    def generate_tic(self):

        """
        Generates a dictionary for the available TOIs and TIC IDs (as to toi. pickle) and
        a dataframe for all TOIs (as to toi_info.pickle)

        Returns
        -------
        duplicate_tic_ids: csv file
            Stores all the duplicate TIC IDs in the current folder
        toi: pickle file
            Stores all the TOIs and TIC IDs as a dict
        toi_info: pickle file
            Stores all the TOIs information from toi.csv

        """

        seen = set()
        dup = list()

        # Create a '/data' folder if it doesn't exists already
        if not os.path.exists('data'):
            os.makedirs('data')

        # Read 'toi.csv' and store it as a pandas dataframe
        data = pd.read_csv(self.path + "/toi.csv", header=0, sep=',')

        # Drop all the irrelevant columns
        data = data.drop(columns=['Previous CTOI', 'Master', 'SG1A', 'SG1B', 'SG2',
       'SG3', 'SG4', 'SG5', 'ACWG ESM', 'ACWG TSM', 'Time Series Observations',
       'Spectroscopy Observations', 'Imaging Observations', 'TESS Disposition',
       'TFOPWG Disposition', 'TESS Mag', 'TESS Mag err', 'Planet Name',
       'Pipeline Signal ID', 'Source', 'Detection','Sectors', 'Date TOI Alerted (UTC)',
       'Date TOI Updated (UTC)', 'Date Modified', 'Comments'])

        # Drop all the duplicate rows if exists
        data = data.drop_duplicates()

        # Convert 'Epoch (BJD)' to 'Epoch (BTJD)' and drop column 'Epoch (BJD)'
        data['Epoch (BTJD)'] = data['Epoch (BJD)'] - 2457000.0
        data = data.drop(columns=['Epoch (BJD)'])

        # Pre-process the columns - TIC ID and TOI
        data['TIC ID'] = 'TIC ' + data['TIC ID'].astype(str)
        data['TOI'] = 'TOI ' + data['TOI'].astype(str)

        # Drop all the rows where Period (days)' == 0
        data = data.loc[(data['Period (days)'] != 0.0)]

        self.toi_info = data

        unique_tic_ids = data['TIC ID'].astype('str').tolist()

        # Find and store all the duplicate TIC IDs
        for x in unique_tic_ids:
            if x in seen:
                dup.append(x)
            else:
                seen.add(x)


        with open(self.path + f"/duplicate_tic_ids.csv", 'w', newline = "") as f:
      
                write = csv.writer(f)
                for i in dup:
                    write.writerow([i])
        f.close()


        toi_info = dict(zip(data['TOI'], data['TIC ID']))
        self.toi = toi_info

        # Store TOIs and TIC IDs
        with open('data/toi.pickle', 'wb') as file:
            pickle.dump(self.toi, file)

        with open('data/toi_info.pickle', 'wb') as file:
            pickle.dump(self.toi_info, file)

        


    def save_folded_lightcurve(self, tois, toi_info, rerun=False):

        """
        Generate pre-processed light curves

        Parameters
        ----------
        tois: dict
            Stores all the TOIs and TIC IDs as key-value pair
        toi_info: pandas-dataframe
            Stores all the TOIs information from toi.csv
        rerun: bool (default=False)
            Generates light curves for TIC IDs which has any missing sector in Lightkurve()

        Notes
        -----
        The light curves generated has 300 data samples.

        Returns
        -------
        Pre-processed light curves as - TOI.csv.gz in /folded_lightcurves

        """

        unsearched_tois=dict()
        searched_tois=dict()
        lcs=list()

        # Create a '/folded_lightcurves' folder if it does not exists already
        if not os.path.exists('folded_lightcurves'):
            os.makedirs('folded_lightcurves')

        # Search all the available TOIs in toi.csv
        for toi in tois:

            try:
                print("\n ID: ",len(searched_tois)," Total: ",len(tois),"\n")
                id_info = toi_info[toi_info['TOI'] == toi].reset_index()
                tic_id, period, epoch, duration_hours, depth = id_info['TIC ID'][0], id_info['Period (days)'][0], id_info['Epoch (BTJD)'][0],id_info['Duration (hours)'][0], id_info['Depth (ppm)'][0]
                fractional_duration = (duration_hours / 24.0)

                l_curves = lk.search_lightcurve(tic_id)

                # Search for authors - SPOC, QLP, TESS-SPOC - for any valid TIC ID
                if l_curves:
                    authors_list=set(l_curves.author)

                    # Search if any author exists for TIC ID
                    if 'SPOC' in authors_list:
                        search_tic_id=l_curves[l_curves.author=='SPOC']
                        min_time, max_time = min(search_tic_id.exptime), max(search_tic_id.exptime)
                        searched_tois[toi]=['SPOC', depth, min_time, max_time, tic_id, duration_hours]

                    elif 'QLP' in authors_list:
                        search_tic_id=l_curves[l_curves.author=='QLP']
                        min_time, max_time = min(search_tic_id.exptime), max(search_tic_id.exptime)
                        searched_tois[toi]=['QLP', depth, min_time, max_time, tic_id, duration_hours]

                    elif 'TESS-SPOC' in authors_list:
                        search_tic_id=l_curves[l_curves.author=='TESS-SPOC']
                        min_time, max_time = min(search_tic_id.exptime), max(search_tic_id.exptime)
                        searched_tois[toi]=['TESS-SPOC', depth, min_time, max_time, tic_id, duration_hours]

                    # If the TIC ID doesn't have a valid author then store it in a dict
                    else:
                        unsearched_tois[toi]=tic_id
                        continue


                    # Download individual light curves due to missing sectors
                    if rerun:
                        if len(search_tic_id) == 1:
                            x = search_tic_id.download()
                            lcs.append(x)
                        else:
                            for i in range(0, len(search_tic_id)):
                                x= search_tic_id[i].download()
                                lcs.append(x)
                        tic_collections = lk.LightCurveCollection(lcs)

                    # Download all the light curves
                    else:
                        tic_collections = search_tic_id.download_all()


                # If there in no data for TIC ID
                else:
                    unsearched_tois[toi]=tic_id
                    continue





                # Pre-process the light curves
                tics = tic_collections.stitch()
                tic_clean=tics.remove_outliers(sigma=20, sigma_upper=4)
                lc_folded = tic_clean.fold(period= period, epoch_time=epoch)
                phase_mask = (lc_folded.phase > -1.50*fractional_duration) & (lc_folded.phase < 1.50*fractional_duration)
                lc_zoom = lc_folded[phase_mask]
                binned_lc= lc_zoom.bin(time_bin_size=fractional_duration/150).normalize() - 1

                # Store the light curves as pandas-dataframe with sorted 'time' column
                binned_lc = binned_lc.to_pandas().reset_index()
                binned_lc = binned_lc.sort_values(by=['time'], ascending=True)


                # Adjust the dataframe size to 300
                bin_size = binned_lc.time.shape[0]

                if bin_size > 300:
                    boundary_points = bin_size - 300
                    q, r = divmod(boundary_points, 2)
                
                    if r == 0:
                        binned_lc = binned_lc[q:-q]
                    else:
                        binned_lc = binned_lc[q:-q-1]

                binned_lc = binned_lc.reset_index(drop=True)

                # Normalize the 'flux' column
                binned_lc.flux = (binned_lc.flux / np.abs(np.nanmin(binned_lc.flux))) * 2.0 + 1

                # Store the columns - time, flux, flux_err
                tic_folded = binned_lc[['time', 'flux', 'flux_err']]
                tic_folded = tic_folded.rename({'time':'phase'}, axis=1)

                # Save the light curves in '/folded_lightcurves'
                lightcurve_path = self.path + f'/folded_lightcurves/{toi}.csv.gz'
                tic_folded.to_csv(path_or_buf=lightcurve_path, header=True, index=False, compression='gzip')
                
                

            except Exception as e:
                unsearched_tois[toi]=tic_id
                print(f"Exception Raised!!\n--- {e}\n\n"f"{toi} --- TIC ID: {tic_id} doesn't have any data\n\n")
                continue


        if unsearched_tois and (not rerun):

            with open(self.path+f"/unsearched_toi.pickle", 'wb') as file:
                pickle.dump(unsearched_tois, file)

            with open('data/tic_info.pickle', 'wb') as file:
                pickle.dump(searched_tois, file)


        elif unsearched_tois and rerun:


            with open(self.path+f"/unresolved_toi.pickle", 'wb') as file:
                pickle.dump(unsearched_tois, file)


            with open('data/tic_info_rerun.pickle', 'wb') as file:
                pickle.dump(searched_tois, file)







    def read_toi_from_pickle(self):

        """
        Returns
        -------
        toi: dict
            Load all the TOIs and TIC IDs from the pickle file
        """

        with open('data/toi.pickle', 'rb') as file:
            toi = pickle.load(file)

        return toi



    def read_toi_info_from_pickle(self):

        """
        Returns
        -------
        toi_info: pandas-dataframe
            Load the toi data from the pickle file
        """

        with open('data/toi_info.pickle', 'rb') as file:
            toi_info = pickle.load(file)

        return toi_info

    def read_unsearched_tois(self):

        """
        Returns
        -------
        unsearched_toi: dict
            Load the unsearched TOIs from the pickle file
        """

        with open(self.path+"/unsearched_toi.pickle", 'rb') as file:
            unsearched_toi = pickle.load(file)

        return unsearched_toi


    def read_data_samples(self):

        """
        Print all the TOIs that have less than 300 data samples and
        Total light curves where >50% of flux values are NaNs
        """
        count = 0
        file_list = os.listdir(self.path+"/folded_lightcurves")
        for file in file_list:
            tics = pd.read_csv(self.path+f"/folded_lightcurves/{file}", compression = 'gzip')
            if (len(tics)!=300):
                print(f"{file[:-7]}: length - {len(tics)}\n")

            count_nans = tics['flux'].isna().sum()
            if (count_nans > 150):
                count += 1
        print(f"Total light curves where >50% of flux values are NaNs: {count}\n")


    def preprocess_lightcurves(self):

        """
        Returns
        -------
        lightcurves: dict
            Contains 'phase' (key) values as a list of length 300 and
            'flux' (key) as ndarray of shape (n, 300) where 'n' is the
            total number of light curves generated.
        """

        file_list = list()
        file_exception = list()
        flux = list()
        lightcurves = dict()
        scaler = StandardScaler()

        with open('data/tic_info.pickle', 'rb') as file:
            tic_info = pickle.load(file)

        tois = list(tic_info.keys())

        # Search all the available TOIs
        for toi in tois:
            file_list.append(toi+'.csv.gz')


        for file in file_list:
            try:
                tics = pd.read_csv(self.path+f"/folded_lightcurves/{file}", compression = 'gzip')
                author, depth, min_time, max_time, tic_id, duration = tic_info[file[:-7]]
                #
                # Store those light curves where the shape is (300,1),
                # Transform the flux using StandardScaler(), and
                # Replace the NaNs with a higher negative value (-999.0)
                #
                if (len(tics.flux) == 300):
                    scaler.fit(tics.flux.values.reshape(-1,1))
                    tics.flux = scaler.transform(tics.flux.values.reshape(-1, 1))
                    tics.flux.fillna(value=-999.0, inplace=True)
                    #
                    # Normalize the phase column
                    #
                    tics.phase = tics.phase/duration
                    flux.append(tics.flux.values.tolist())

            except Exception as e:
                print(f"\nException Raised: {e}\n"f"Check : {file[:-7]}, {file}")
                #
                # Store the TOIs for which the flux couldn't be stored
                #
                file_exception.append(file[:-7])
                continue

        #
        # Store the phase and flux values in the dictionary
        #
        flux = np.array(flux)
        lightcurves['phase'] = tics.phase.values.tolist()
        lightcurves['flux'] = flux


        # Store the files
        with open(self.path+f"/data/lightcurves.pickle", 'wb') as file:
            pickle.dump(lightcurves, file)

        if file_exception:
            with open(self.path+f"/file_exception.pickle", 'wb') as file:
                pickle.dump(file_exception, file)
            print("\n Flux values for some TOIs couldn't be stored.\n Please refer file_exception.pickle file!")





    def generate_tic_image(self, folder=None):

        """
         Generates images for all the available TOIs in '/images'
        """

        file_list = list()

        # Create a '/images' folder if it does not exists already
        if not os.path.exists('images'):
            os.makedirs('images')

        # Load the tic_info.pickle file
        with open('data/tic_info.pickle', 'rb') as file:
            tic_info = pickle.load(file)

        # Sort the TOIs wrt its depth (ascending order)
        sorted_tic_info = OrderedDict(sorted(tic_info.items(), key=lambda x: x[1][1]))
        tois = list(sorted_tic_info.keys())

        # Search all the available TOIs
        for toi in tois:
            file_list.append(toi+'.csv.gz')

        # Generate the images
        n_row, n_col, id = 8, 3, 0
        q, r = divmod(len(tois), n_row*n_col)
        if r != 0: q += 1
        batch_size = q
        
        for batch in range(batch_size):
            fig = plt.figure(figsize=(1024, 2048))
            fig, axs= plt.subplots(nrows = n_row, ncols = n_col, figsize=(20,15))

            k = id
            for i in range(n_row):
        
                for j in range(n_col):
                    
                    try:
                        if id < len(tois):
                            toi = tois[id]
                            author, depth, min_time, max_time, tic_id, duration = sorted_tic_info[toi]
                            axs[i,j].set_title(f"{toi} -- Depth:{depth:.2f} ppm -- Exp_Time: {max_time} -- Transit: {duration:.2f} hrs")
                            binned_lc = pd.read_csv(self.path+f"/{folder}/{file_list[id]}", compression = 'gzip')
                            x = np.array(binned_lc['phase'])
                            y = np.array(binned_lc['flux'])
                            axs[i,j].scatter(x,y, s=5, c='black', label=f"{tic_id}--{author}")
                            axs[i,j].legend(loc='lower left')
            
                            id += 1
                
                    except Exception as e:
                        print(f"Exception Raised: {e}\n\n"f"Validate TOI: {toi}, TIC ID: {tic_id}")
                        id += 1
                        continue
                    
            fig.tight_layout(pad=1.0)
            fig.savefig(self.path+f"/images/image_{k}_{id-1}.png", bbox_inches ="tight", orientation ='landscape')
        
        



    
if __name__ == '__main__':


    create_lightcurves = TESS_Lightcurves()

    create_lightcurves.generate_tic()
    tois = create_lightcurves.read_toi_from_pickle()
    toi_info = create_lightcurves.read_toi_info_from_pickle()
    #
    create_lightcurves.save_folded_lightcurve(tois, toi_info, rerun=False)
    print("\n\nLight curves are generated!\n")
    #
    unsearched_tic_ids = create_lightcurves.read_unsearched_tois()
    create_lightcurves.save_folded_lightcurve(unsearched_tic_ids, toi_info, rerun=True)
    print("\n\nUnsearched light curves are generated!\n")
    #
    print("TOIs below 300 data samples: \n")
    create_lightcurves.read_data_samples()
    #
    create_lightcurves.generate_tic_image(folder='folded_lightcurves')
    print("Images are available!\n")
    #
    create_lightcurves.preprocess_lightcurves()
    print("light curves are pre-processed and stored in /data!\n")


