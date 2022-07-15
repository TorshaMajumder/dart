import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def read_data_samples():
    with open('data/lightcurves.pickle', 'rb') as file:
        lightcurves = pickle.load(file)

    filtered_lcs = lightcurves['pca']
    flux = filtered_lcs['flux_pca']

    return flux



class UnsupervisedRandomForest(object):

    def __init__(self, X = None, n_features = None, max_depth = None, min_samples_split = 2,
                 min_samples_leaf = 1, max_features = None, bootstrap = False, min_impurity_decrease=0.0,
                 max_samples = None, n_samples = None, n_estimators = 100, random_state = 0, criterion="gini"):

        self.path = os.path.dirname(os.path.abspath(__file__))
        self.X = X
        self.n_features = n_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_samples = n_samples
        self.n_estimators = n_estimators
        self.estimators = None
        self.random_state = random_state






    def generate_data(self):

        syn_flux = np.zeros(self.X.shape)

        for i in range(self.n_features):
            feature_vec = self.X[:,i]
            syn_feature_vec = np.random.choice(feature_vec, self.n_samples)
            syn_flux[:,i] += syn_feature_vec


        Y_real = np.ones(self.n_samples)
        Y_syn = np.ones(len(syn_flux)) * 2

        Y = np.concatenate((Y_real, Y_syn))
        X = np.concatenate((self.X, syn_flux))

        return X, Y



    def fit(self, X, y):

        self.estimators = RandomForestClassifier(n_estimators=self.n_estimators,
                                max_features=self.max_features, max_depth=self.max_depth,
                                min_impurity_decrease=self.min_impurity_decrease,
                                min_samples_split=self.min_samples_split,
                                bootstrap=self.bootstrap, random_state=self.random_state)

        self.estimators.fit(X,y)



    def dissimilarity_matrix(self, plot=False):

        leaf_index = self.estimators.apply(self.X)
        print(leaf_index.shape)
        class_matrix = np.zeros(leaf_index.shape)

        for i, estimator in enumerate(self.estimators.estimators_):

            mask_real_data = estimator.predict_proba(self.X)[:, 0] == 1
            class_matrix[:, i] = mask_real_data

        leaf_index[class_matrix == False] = -1
        leaf_count=np.zeros((np.max(leaf_index[:,:]+1),self.n_estimators))

        for i in range(self.n_samples):
            for j in range(self.n_estimators):
                if leaf_index[i,j]!= -1:
                    leaf_count[leaf_index[i,j],j]+=1



        normalized_leaf_count=np.sum(leaf_count,axis=0)

        sim_matrix=np.zeros(self.n_samples)
        for i in range(self.n_samples):
            for j in range(self.n_estimators):

                sim_matrix[i]+=(leaf_count[(leaf_index[i,j]),j])/normalized_leaf_count[j]


        sim_matrix/= self.n_estimators
        dis_matrix = 1 - sim_matrix

        if plot:

            plt.figure(figsize=(10, 5))
            plt.title("URF Anomaly Score")
            fig = plt.hist(dis_matrix, bins=30, color="yellowgreen", edgecolor='black', linewidth=1.2)
            plt.ylabel("#Lightcurves", size=12)
            plt.xlabel("Anomaly Score", size=12)
            plt.show()


        return dis_matrix



    def get_feature_importance(self, plot=False):

        estimators = SelectFromModel(self.estimators, prefit=True)
        selected_features = estimators.get_support()
        features_score=self.estimators.feature_importances_
        features = np.arange(self.n_features)
        feature_set = [y for x,y in zip(selected_features, features) if x]
        feature_set_score=features_score[selected_features]
        sorted_set = sorted(zip(feature_set, feature_set_score), key=lambda x: x[1], reverse=True)

        if plot:

            # feature_id = np.arange(self.n_features)
            norm_feature_score = features_score/np.max(features_score)

            plt.figure(figsize=(15, 5))
            plt.title("URF Feature Importance")
            cm = plt.cm.get_cmap('viridis')
            fig = plt.scatter(features,norm_feature_score,c=norm_feature_score,cmap=cm)
            fig.axes.yaxis.set_ticklabels([])
            plt.xlim(0,self.n_features)
            plt.ylim(0, 1.1)
            plt.xlabel('Feature ID',size=12)
            cbar = plt.colorbar(fig,pad=0.05)
            cbar.set_label('Feature Importance',size=12,rotation=270,labelpad=17)
            plt.show()


        return sorted_set





if __name__ == '__main__':

    X_train = read_data_samples()

    params = { 'X': X_train,
              'n_features': X_train.shape[1],
              'max_depth': 100,
              'min_samples_split': 3,
              'max_features': 'log2',
              'bootstrap': False,
              'n_samples': X_train.shape[0],
              'n_estimators': 50,
              'random_state': 0
            }

    URF = UnsupervisedRandomForest(**params)
    X, y = URF.generate_data()
    URF.fit(X,y)
    dis_matrix = URF.dissimilarity_matrix(plot=True)
    features = URF.get_feature_importance(plot=True)



