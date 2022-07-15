import tsfresh as tf
import pickle
import os
import numpy as np
from TESS.load_data import load_data_tsfresh
from TESS.anomaly_detection.UnsupervisedRF import UnsupervisedRandomForest




class TSFresh(object):
    def __init__(self, X=None, n_features=10):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.X= X
        self.n_features = n_features


    def extract_features(self):

        extracted_features = tf.extract_features(self.X, column_id='TOI', column_sort='PHASE')
        extracted_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        extracted_features.fillna(99999, inplace=True)
        #
        # Store the files
        #
        with open(f"../data/tsfresh_data.pickle", 'wb') as file:
            pickle.dump(extracted_features, file)


    def get_important_features(self):

        feature_list = list()

        with open('../data/tsfresh_data.pickle', 'rb') as file:
            extracted_features = pickle.load(file)


        features_name = extracted_features.columns
        extracted_features = extracted_features.to_numpy()

        params = { 'X': extracted_features,
                   'n_features': extracted_features.shape[1],
                   'max_depth': 100,
                   'min_samples_split': 3,
                   'max_features': 'log2',
                   'bootstrap': False,
                   'n_samples': extracted_features.shape[0],
                   'n_estimators': 100,
                   'random_state': 0
                   }

        URF = UnsupervisedRandomForest(**params)
        X, y = URF.generate_data()
        URF.fit(X,y)
        features = URF.get_feature_importance(plot=False)

        for i in range(self.n_features):
            idx, val = features[i][0], features[i][1]
            feature_list.append(idx)
        print(feature_list, features_name[feature_list])





if __name__ == '__main__':

    X_train = load_data_tsfresh()
    params={'X': X_train,
            'n_features':10
            }

    tsfresh = TSFresh(**params)
    #tsfresh.extract_features()
    tsfresh.get_important_features()





