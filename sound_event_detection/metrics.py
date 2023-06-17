import numpy as np
import pandas as pd
import sed_eval
import sklearn.metrics as skmetrics

from utils import encode_label


def get_audio_tagging(data):
    return data.groupby('filename')['event_label'].unique().reset_index()


def get_audio_tagging_results(reference, estimated, label_to_idx):
    def nan_value(value):
        if isinstance(value, (np.ndarray, list)):
            return np.array(value)
        if pd.isna(value):
            return np.zeros(len(label_to_idx))
        return value
        
    if "event_label" in reference.columns:
        reference = get_audio_tagging(reference)
        estimated = get_audio_tagging(estimated)
        ref_labels = reference["event_label"].apply(lambda x: encode_label(x, label_to_idx))
        reference['event_label'] = ref_labels
        est_labels = estimated["event_label"].apply(lambda x: encode_label(x, label_to_idx))
        estimated['event_label'] = est_labels
        
    matching = reference.merge(estimated, how='outer', on="filename", suffixes=["_ref", "_pred"])
    
    result_data = pd.DataFrame(columns=['label', 'f1', 'precision', 'recall'])
    
    matching['event_label_pred'] = matching.event_label_pred.apply(nan_value)
    matching['event_label_ref'] = matching.event_label_ref.apply(nan_value)
    y_true = np.vstack(matching['event_label_ref'].values)
    y_pred = np.vstack(matching['event_label_pred'].values)
    result_data.loc[:, 'label'] = label_to_idx.keys()
    average_types = [None, 'macro', 'micro']
    for avg in average_types:
        avg_f1 = skmetrics.f1_score(y_true, y_pred, average=avg, zero_division=0)
        avg_pre = skmetrics.precision_score(y_true, y_pred, average=avg, zero_division=0)
        avg_rec = skmetrics.recall_score(y_true, y_pred, average=avg, zero_division=0)
        
        if avg is None:
            result_data['precision'] = avg_pre
            result_data['recall'] = avg_rec
            result_data['f1'] = avg_f1
        else:
            result_data = pd.concat([
                result_data, 
                pd.DataFrame({
                    "label": avg,
                    "precision": avg_pre,
                    "recall": avg_rec,
                    "f1": avg_f1
                }, index=[0])
            ], ignore_index=True)
    return result_data


def get_event_list(data, filename):
    event_file = data[data['filename'] == filename]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list= [{"filename": filename}]
        else:
            event_list = event_file.to_dict('records')
    else:
        event_list = event_file.to_dict('records')
    return event_list


def event_based_evaluation(reference, estimated, t_collar=0.2, percentage_of_length=0.2):
    evaluation_files = reference['filename'].unique()
    
    classes = []
    classes.extend(reference['event_label'].dropna().unique())
    classes.extend(estimated['event_label'].dropna().unique())
    classes = list(set(classes))
    
    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score'
    )
    for filename in evaluation_files:
        reference_event_list = get_event_list(reference, filename)
        estimated_event_list = get_event_list(estimated, filename)
        event_based_metric.evaluate(
            reference_event_list=reference_event_list,
            estimated_event_list=estimated_event_list
        )
    return event_based_metric


def segment_based_evaluation(reference, estimated, time_resolution=1.0):
    evaluation_files = reference['filename'].unique()
    
    classes = []
    classes.extend(reference['event_label'].dropna().unique())
    classes.extend(estimated['event_label'].dropna().unique())
    classes = list(set(classes))
    
    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=time_resolution
    )
    for filename in evaluation_files:
        reference_event_list = get_event_list(reference, filename)
        estimated_event_list = get_event_list(estimated, filename)
        segment_based_metric.evaluate(
            reference_event_list=reference_event_list,
            estimated_event_list=estimated_event_list
        )
    return segment_based_metric


def compute_metrics(reference, estimated):
    metric_event = event_based_evaluation(reference, estimated)
    metric_segment = segment_based_evaluation(reference, estimated)
    return metric_event, metric_segment
