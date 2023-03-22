# An example for the CRTS data
from astronomaly.data_management import light_curve_reader
from astronomaly.feature_extraction import feets_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import svm, human_loop_learning, isolation_forest
from astronomaly.visualisation import umap_plot
import os
import pandas as pd
import pickle


# Root directory for data
data_dir = os.path.join(os.getcwd(), 'example_data')
lc_path = os.path.join(data_dir, 'CRTS', 'data', 'cluster_id_-1')
umap_data = os.path.join(data_dir, 'CRTS', 'umap_features')
tsfresh_data = os.path.join(data_dir, 'CRTS', 'tsfresh_aad.pickle')

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'CRTS', '')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

display_transform_function = []
# Change this to false to automatically use previously run features
force_rerun = True 


def run_pipeline():
    """
    Any script passed to the Astronomaly server must implement this function.
    run_pipeline must return a dictionary that contains the keys listed below.

    Parameters
    ----------

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data. Keys must include:
        'dataset' - an astronomaly Dataset object
        'features' - pd.DataFrame containing the features
        'anomaly_scores' - pd.DataFrame with a column 'score' with the anomaly
        scores
        'visualisation' - pd.DataFrame with two columns for visualisation
        (e.g. TSNE or UMAP)
        'active_learning' - an object that inherits from BasePipeline and will
        run the human-in-the-loop learning when requested

    """

    # This creates the object that manages the data
    lc_dataset = light_curve_reader.LightCurveDataset(
        directory=lc_path,
        data_dict={'id': 7, 'time': 0, 'flux': [1, 3], 'flux_err': [2, 4]},
        filter_labels=['g', 'r'],
        output_dir=output_dir
    )

    # Extracted features
    # with open(f"{tsfresh_data}", 'rb') as file:
    #     features = pickle.load(file)
    features = pd.read_csv(f"{umap_data}/cluster_id_-1.csv")
    features = features.set_index("Unnamed: 0")
    features.index.name = None

    # The actual anomaly detection is called in the same way by creating an
    # OneClassSVM pipeline object then running it
    pipeline_svm = svm.OneClassSVM_Algorithm(
        force_rerun=force_rerun, output_dir=output_dir)
    anomalies = pipeline_svm.run(features)
    # pipeline_iforest = isolation_forest.IforestAlgorithm(
    #     force_rerun=force_rerun, output_dir=output_dir)
    # anomalies = pipeline_iforest.run(features)

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=force_rerun, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)

    try:
        # This is used by the frontend to store labels as they are applied so
        # that labels are not forgotten between sessions of using Astronomaly
        if 'human_label' not in anomalies.columns:
            df = pd.read_csv(
                os.path.join(output_dir, 'ml_scores.csv'),
                index_col=0,
                dtype={'human_label': 'int'})
            df.index = df.index.astype('str')

            if len(anomalies) == len(df):
                anomalies = pd.concat(
                    (anomalies, df['human_label']), axis=1, join='inner')
    except FileNotFoundError:
        pass

    # This is the active learning object that will be run on demand by the
    # frontend
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir)

    # We use UMAP for visualisation which is run in the same way as other parts
    # of the pipeline.
    pipeline_umap = umap_plot.UMAP_Plot(
        force_rerun=False,
        random_state=42,
        output_dir=output_dir)
    u_plot = pipeline_umap.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': lc_dataset,
            'features': features,
            'anomaly_scores': anomalies,
            'visualisation': u_plot,
            'active_learning': pipeline_active_learning}


if __name__ == '__main__':

    #run_pipeline()
    plot = pd.read_parquet('/Users/june/git/astronomaly/astronomaly/scripts/example_data/astronomaly_output/CRTS/UMAP_Plot_output.parquet', engine='pyarrow')
    print(plot)


