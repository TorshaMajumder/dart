#
# Import all the dependencies
#
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from TESS.datasets.transients import load_latent_space

#
# Create '/results/anomaly_detection' folder if it does not exists already
#
if not os.path.exists('../results/anomaly_detection'):
    os.makedirs('../results/anomaly_detection')

class UnsupervisedRandomForest(object):

    """
    Unsupervised Random Forest is an anomaly detection technique
    used for the light curves from NASA's TESS telescope.

    The anomaly scores are available in the folder -
        -- ../results/anomaly_detection/transients
        -- ../results/anomaly_detection/transits

    Parameter
    --------
    n_features: int
        number of features in the data sample

    criterion: {“gini”, “entropy”, “log_loss”} (default=”gini”)
        function to measure the quality of a split

    max_depth: int
        maximum depth of the tree

    min_samples_split: int or float (default=2)
        minimum number of samples required to split an internal node

    min_samples_leaf: int or float (default=1)
        minimum number of samples required to be at a leaf node

    max_features: int or float (default = 1.0)
        number of features to draw from X to train each base estimator

    bootstrap: bool (default=False)
        if True, individual trees are fit on random subsets of the training
        data sampled with replacement; if False, sampling without replacement is performed

    max_samples: “auto”, int or float (default=256)
        number of samples to draw from X to train each base estimator

    n_samples: int or float
        number of data samples

    n_estimators: int (default=100)
        number of base estimators in the ensemble

    random_state: int (default=0)
        controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest

    contamination: float (default=0.1)
        amount of contamination of the data set, i.e. the proportion of outliers in the data set

    labels: string
        IAU Name or TIC ID

    lc_type: string
        'transients' or 'transits'

    extract_type: string
        feature extraction type - 'tsfresh', 'vae', and 'k_pca'


    """


    def __init__(self, X=None, n_features=None, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, bootstrap=False,
                 max_samples=None, n_samples=None, n_estimators=100, random_state=0,
                 criterion="gini", labels=None, lc_type=None, extract_type=None, contamination=0.1):

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.X = X
        self.n_features = n_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_samples = n_samples
        self.n_estimators = n_estimators
        self.estimators = None
        self.random_state = random_state
        self.labels = labels
        self.type = lc_type
        self.extract_type = extract_type
        self.anomaly_score = None
        self.anomaly_index = None
        self.contamination = contamination

        try:
            if self.type not in ["transits", "transients"]:
                raise TypeError(f"\nTypeError: '{self.type}' is not a valid type!"
                                f"\nPlease provide the type as - 'transits' or 'transients'")
        except Exception as e:
            print(e)
            exit()

        try:
            if self.extract_type not in ["k_pca", "tsfresh", "vae"]:
                raise TypeError(f"\nTypeError: '{self.extract_type}' is not a valid type!"
                                f"\nPlease provide the type as - 'k_pca' , 'tsfresh', or 'vae'")
        except Exception as e:
            print(e)
            exit()

    def generate_data(self):

        """
        This function returns a matrix with the same dimensions as X but with synthetic data
        based on the marginal distributions of its features
        """

        syn_flux = np.zeros(self.X.shape)
        #
        # Get the synthetic data
        #
        for i in range(self.n_features):
            feature_vec = self.X[:,i]
            syn_feature_vec = np.random.choice(feature_vec, self.n_samples)
            syn_flux[:,i] += syn_feature_vec
        #
        # Give label "1" to the real data and label "2" to the synthetic data
        #
        Y_real = np.ones(self.n_samples)
        Y_syn = np.ones(len(syn_flux)) * 2
        #
        # Merge the data into one sample
        #
        Y = np.concatenate((Y_real, Y_syn))
        X = np.concatenate((self.X, syn_flux))

        return X, Y



    def fit_transform(self):

        """
        Fits and transforms the data using Random Forest classifier to measure the weirdness

        Returns
        -------
        anomaly_index: ndarray
            index of the anomalies

        anomaly_score: ndarray
            anomaly score of the data samples (negative scores for anomalies)
        """
        #
        # Initialize a weirdness matrix for 'N' runs
        #
        weirdness = np.zeros((self.n_samples, 2))
        #
        # Number of runs
        #
        runs = 10
        #
        # Select the (100-(idx*100))th percentile of the weirdness score
        #
        idx = int(self.n_samples*self.contamination)
        #
        #
        #
        for i in range(runs):
            #
            # Generate data for each run
            #
            X, y = self.generate_data()
            #
            # Declare a Random Forest classifier
            #
            est = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features,
                                         max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,bootstrap=self.bootstrap,
                                         random_state=self.random_state)
            #
            # Fit data to a Random Forest classifier
            #
            est.fit(X, y)
            #
            # Build the dissimilarity matrix
            #
            dis_mat = self.dissimilarity_matrix(est, X)
            #
            # Calculate the average weirdness score
            #
            for j in range(self.n_samples):
                weirdness[j][0] += dis_mat[j]
                weirdness[j][1] = int(j)

                if i == (runs-1):
                    weirdness[j][0] /= runs
        #
        # Sort the weirdness score
        #
        sorted_weirdness = weirdness[weirdness[:,0].argsort()]
        #
        # Store the anomaly index and score
        #
        self.anomaly_index = sorted_weirdness[:, 1]
        self.anomaly_index = self.anomaly_index.astype(np.int64)
        self.anomaly_index = self.anomaly_index[-idx:]
        self.anomaly_score = weirdness[:, 0]
        #
        # Prepare a dictionary using IAU Name, anomaly index, and anomaly score
        #
        anomaly_labels = {'labels': self.labels, 'anomaly_index': self.anomaly_index,
                          'anomaly_score': self.anomaly_score}
        #
        # Create '/results/clustering/{type}/' folder if it does not exists already
        #
        if not os.path.exists(f"../results/anomaly_detection/{self.type}"):
            os.makedirs(f"../results/anomaly_detection/{self.type}")
        #
        # Store the file in -- '/results/clustering/{type}/' folder
        #
        with open(f"../results/anomaly_detection/{self.type}/urf_{self.extract_type}.pickle", 'wb') as file:
            pickle.dump(anomaly_labels, file)
        #
        #
        #
        print(f"\nAnomaly scores are generated and stored "
              f"in -- /results/anomaly_detection/{self.type} -- folder!\n")

        return self.anomaly_index, self.anomaly_score



    def fit(self, X, y):

        """
        Fits the data using Random Forest classifier (used in TSFresh feature extraction method)
        """

        self.estimators = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features,
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 bootstrap=self.bootstrap,
                                                 random_state=self.random_state)

        self.estimators.fit(X, y)



    def dissimilarity_matrix(self, est, X):

        """
        Get the anomaly score using Unsupervised Random Forest

        This function builds the similarity matrix based on the feature matrix X
        for the results Y based on the trained random forest classifier. The matrix
        is normalised so that the biggest similarity is 1 and the lowest is 0.

        This function counts only leaves in which the object is classified as a "real" object
        it is also implemented to optimize running time, assuming one has enough running memory.


        """
        #
        # Apply to get the leaf indices
        #
        leaf_index = est.apply(X)
        #
        # Find the predictions of the sample
        #
        class_matrix = np.zeros(leaf_index.shape)
        for i, estimator in enumerate(est.estimators_):
            mask_real_data = estimator.predict_proba(X)[:, 0] == 1
            class_matrix[:, i] = mask_real_data
        #
        # Mark leaves that make the wrong prediction as -1, in order
        # to remove them from the distance measurement
        #
        leaf_index[class_matrix == False] = -1
        leaf_count=np.zeros((np.max(leaf_index[:,:]+1),self.n_estimators))
        #
        # Creates an array of each leaf population for all trees in RF
        #
        for i in range(self.n_samples):
            for j in range(self.n_estimators):
                if leaf_index[i,j]!= -1:
                    leaf_count[leaf_index[i,j],j]+=1
        #
        # Used to normalize and weight the results heavily towards trees that
        # correctly identify objects as real or synthetic
        #
        normalized_leaf_count=np.sum(leaf_count,axis=0)
        sim_matrix=np.zeros(self.n_samples)
        #
        # Reduces each population by 1 to avoid self comparison
        #
        for i in range(self.n_samples):
            for j in range(self.n_estimators):
                sim_matrix[i]+=(leaf_count[(leaf_index[i,j]),j])/normalized_leaf_count[j]
        #
        # Creates the similarity matrix by using the leaf populations
        # rather then pair matching each light curve
        #
        sim_matrix /= self.n_estimators
        self.anomaly_score = 1 - sim_matrix

        return self.anomaly_score



    def get_feature_importance(self):

        """
        Get the feature importance using Isolation Forest (used in TSFresh feature extraction method)
        """
        X, y = self.generate_data()
        self.fit(X, y)
        estimators = SelectFromModel(self.estimators, prefit=True)
        selected_features = estimators.get_support()
        features_score=self.estimators.feature_importances_
        features = np.arange(self.n_features)
        feature_set = [y for x,y in zip(selected_features, features) if x]
        feature_set_score=features_score[selected_features]
        sorted_set = sorted(zip(feature_set, feature_set_score), key=lambda x: x[1], reverse=True)
        return sorted_set


if __name__ == '__main__':

    data = load_latent_space(extract_type='vae')
    X_train, labels = data['data'], data['labels']

    params = { 'X': X_train,
              'n_features': X_train.shape[1],
              'max_depth': 100,
              'min_samples_split': 3,
              'max_features': 'log2',
              'bootstrap': False,
              'n_samples': X_train.shape[0],
              'n_estimators': 50,
              'random_state': 0,
              'labels': labels,
              'lc_type': 'transients',
              'contamination': 0.1,
              'extract_type': 'vae'
            }

    URF = UnsupervisedRandomForest(**params)
    anomaly_index, anomaly_score = URF.fit_transform()
    features = URF.get_feature_importance()





