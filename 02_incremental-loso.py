import os
import csv
import argparse
import gc

from alive_progress import alive_bar

from functions.data_loading import load_data
from functions.data_grouping import generate_lno_group, generate_training_and_test_sets
from functions.training import create_trainer
from functions.training_report import generate_report, report_writer


DATA_DIR = '01_DATA'
WINDOWED_DATA_DIR = os.path.join(DATA_DIR, '03_WINDOWED')
MODEL_REPORTS_DIR = '02_MODEL-REPORTS'
REPORTS_PATH = os.path.join(MODEL_REPORTS_DIR, '{0}_models.csv')
LOSO_GROUPS_PATH = os.path.join(MODEL_REPORTS_DIR, 'loso_groups.csv')

ACTIVITIES = {"SEATED": 0, "STANDING_UP": 1, "WALKING": 2, "TURNING": 3, "SITTING_DOWN": 4}

BATCH_SIZE = 20
EPOCHS = 50
N_SPLITS = 10


def train_models(data, subjects, test_subjects, batch_size, epochs, n_splits, testing_mode):
    trainer = create_trainer(batch_size, epochs)
    writers = {}
    loso_writer = loso_group_writer()

    for test_subject in test_subjects:
        with alive_bar(len(subjects) - 1, dual_line=True, title=f'Evaluating models with {test_subject}', force_tty=True) as progress_bar:
            for n in range(1, len(subjects)):
                for i in range(n_splits):
                    train_subjects = generate_lno_group(subjects, n, test_subject)
                    loso_writer(test_subject, n, i, train_subjects)
                    
                    progress_bar.text = f'Training {i+1}th model with {n} subjects'
                    for source, (x, y) in data.items():
                        x_train, y_train, x_test, y_test = generate_training_and_test_sets(x, y, train_subjects, [test_subject])

                        model, training_time = trainer(x_train, y_train, verbose=0)
                        y_pred = model.predict(x_test, verbose=0)

                        report = generate_report(y_test, y_pred, ACTIVITIES.keys())
                        report['training time'] = training_time

                        if not testing_mode:
                            if not source in writers:
                                writers[source] = report_writer(REPORTS_PATH.format(source))
                            writers[source](test_subject, n, i+1, report)
                            
                        del model
                        del x_train
                        del y_train
                        del x_test
                        del y_test
                        
                gc.collect()
                progress_bar()

    
def loso_group_writer():
    fields = ['test_subject', 'n', 'i', 'train_subjects']
    if not os.path.exists(LOSO_GROUPS_PATH):
        with open(LOSO_GROUPS_PATH, mode='w') as loso_csv:
            csv_writer = csv.DictWriter(loso_csv, fieldnames=fields)
            csv_writer.writeheader()
            
    def writer(test_subject, n, i, train_subjects):        
        with open(LOSO_GROUPS_PATH, mode='a') as loso_csv:
            csv_writer = csv.DictWriter(loso_csv, fieldnames=fields)
            csv_writer.writerow({'test_subject': test_subject, 'n': n, 'i': i, 'train_subjects': train_subjects})
            
    return writer
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', help='evaluate only with specified subject', type=int)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', help='training epochs', type=int, default=EPOCHS)
    parser.add_argument('--splits', help='models trained for each case', type=int, default=N_SPLITS)
    parser.add_argument('--testing_script', help='Testing the script. Results not stored', action='store_true')
    args = parser.parse_args()

    x_sp, y_sp = load_data(WINDOWED_DATA_DIR, 'sp', ACTIVITIES)
    x_sw, y_sw = load_data(WINDOWED_DATA_DIR, 'sw', ACTIVITIES)
    data = {
        'sp': (x_sp, y_sp),
        'sw': (x_sw, y_sw)
    }
    subjects = list(x_sp.keys())
    test_subjects = [subjects[args.subject - 1]] if args.subject else subjects
    train_models(data, subjects, test_subjects, args.batch_size, args.epochs, args.splits, args.testing_script)    
