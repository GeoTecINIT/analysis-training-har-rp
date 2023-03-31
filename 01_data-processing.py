import os
import numpy as np
from alive_progress import alive_bar

from functions.raw_data_loading import load_raw_data_and_segments, load_subjects_info
from functions.raw_data_processing import split, temporal_trim, merge, add_labels, scale, windows, compute_best_class, count_data


DATA_DIR = "01_DATA"
RAW_DATA_DIR = os.path.join(DATA_DIR, "01_RAW")
CLEAN_DATA_DIR = os.path.join(DATA_DIR, "02_CLEAN")
WINDOWED_DATA_DIR = os.path.join(DATA_DIR, "03_WINDOWED")

WINDOW_SIZE = 50
STEP_SIZE = WINDOW_SIZE // 2


def clean_raw_data(raw_data, segments):
    clean_data = {}

    with alive_bar(len(raw_data), title=f'Data cleanning', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in raw_data.items():
            execution_name, source = desc.rsplit('_', 1)
            execution_segments = segments.loc[execution_name]
            df_acc, df_gyro = split(data)
            df_acc, df_gyro = temporal_trim(df_acc, df_gyro, execution_segments)    
            df_merged = merge(df_acc, df_gyro)
            df_labeled = add_labels(df_merged, execution_segments)
            clean_data[desc] = scale(df_labeled, source)
            if len(clean_data[desc]) < 100:
                print(f'WARNING on {desc}')
            progress_bar()
            
    return clean_data


def get_windowed_data(clean_data, window_size, step_size): 
    windowed_data = {}
    gt = {}
    
    with alive_bar(len(clean_data), title=f'Data windowing', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in clean_data.items():
            desc_components = desc.split('_')
            subject_sensor_desc = f'{desc_components[0]}_{desc_components[2]}'

            windowed_df = windows(data, window_size, step_size)
            desc_instances = []
            desc_gt = []

            for i in range(0, data.shape[0], step_size):
                window = windowed_df.loc["{0}:{1}".format(i, i+window_size)]
                values = window[['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro']].transpose()
                groundtruth = compute_best_class(window)
                if (values.shape[1] != window_size):
                    break
                desc_instances.append(values.values.tolist())
                desc_gt.append(groundtruth.values[0])

            if subject_sensor_desc in windowed_data:
                windowed_data[subject_sensor_desc] += desc_instances
                gt[subject_sensor_desc] += desc_gt
            else:
                windowed_data[subject_sensor_desc] = desc_instances
                gt[subject_sensor_desc] = desc_gt
                
            progress_bar()
            
    return windowed_data, gt


def store_clean_data(clean_data, path):
    with alive_bar(len(clean_data), title=f'Storing clean data in {path}', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in clean_data.items():
            subject = desc.split('_')[0]
            subject_path = os.path.join(path, subject)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
            data.to_csv(os.path.join(subject_path, f'{desc}.csv'), index=False)
            progress_bar()
        
        
def store_windowed_data(windowed_data, ground_truth, path):
    def store_as_npy(path, data):
        with open(path, 'wb') as f:
            np.save(f, np.array(data)) 
            
    with alive_bar(len(windowed_data), title=f'Storing windowed data in {path}', force_tty=True, monitor='[{percent:.0%}]') as progress_bar:
        for desc, data in windowed_data.items():
            subject = desc.split('_')[0]
            subject_path = os.path.join(path, subject)
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)

            store_as_npy(os.path.join(subject_path, f'{desc}.npy'), data)
            store_as_npy(os.path.join(subject_path, f'{desc}_gt.npy'), ground_truth[desc])
            progress_bar()  


if __name__ == '__main__':
    subjects_info, age_info, gender_info = load_subjects_info(os.path.join(RAW_DATA_DIR, 'subjects.csv'))
    print(age_info)
    print(gender_info, '\n')
    
    raw_data, segments = load_raw_data_and_segments(RAW_DATA_DIR)
    
    clean_data =  clean_raw_data(raw_data, segments)
    print('\nClean data:')
    print(count_data(clean_data), '\n')
    store_clean_data(clean_data, CLEAN_DATA_DIR)
    print('\n')
    
    
    windowed_data, gt = get_windowed_data(clean_data, WINDOW_SIZE, STEP_SIZE)
    print('\nWindowed data:')
    print(count_data(gt), '\n')
    store_windowed_data(windowed_data, gt, WINDOWED_DATA_DIR)
        