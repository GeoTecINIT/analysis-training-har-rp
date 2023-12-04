# Reproducible Package for _"Analysis and Impact of Training Set Size in Cross-Subject Human Activity Recognition"_

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/GeoTecINIT/analysis-training-har-rp/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/zenodo/10.5281/zenodo.8163542/)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.1007%2F978--3--031--49018--7__28-yellow.svg)](https://doi.org/10.1007/978-3-031-49018-7_28)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8163542.svg)](https://doi.org/10.5281/zenodo.8163542)

This repository is the reproducibility package for the _“Analysis and Impact of Training Set Size in Cross-Subject Human Activity Recognition"_ conference paper.

> Matey-Sanz, M., Torres-Sospedra, J., González-Pérez, A., Casteleyn, S., & Granell, C. (2023, November). Analysis and Impact of Training Set Size in Cross-Subject Human Activity Recognition. In Iberoamerican Congress on Pattern Recognition (pp. 391-405). Cham: Springer Nature Switzerland.

You can explore the data and code used to obtain the results presented in the paper. To properly view the Jupyter Notebook files with their rendered figures, click on the "nbviewer" badge above.

## Reproducibility 

### Reproduce online 

Click the on the "Binder" badge above to open an interactive Jupyter environment with all required software installed.

### Reproduce locally
Install Python 3.9, download the repository, open a command line in the root of the directory and install the required software by executing:

```bash
pip install -r requirements.txt
```

### Reproduce locally with Docker
Install [Docker](https://www.docker.com) for building an image based on a `Dockerfile` with a Jupyter environment and running a container based on the image.

Download the repository, open a command line in the root of the directory and:

1. Build the image:

```bash
docker build . --tag analysis-training-har-rp
```

2. Run the image:

```bash
docker run -it -p 8888:8888 analysis-training-har-rp
```

3. Click on the login link (or copy and paste in the browser) shown in the console to access to a Jupyter environment.

### Reproduce the analysis
Execute a script or open the desired Jupyter Notebook (\*.ipynb) file. The notebook contains the code used for the analysis and its outputs. You can execute the code to reproduce the obtained results presented in the paper.

> **Note**: when executing code with a component of randomness (i.e., ML models training), the obtained results could be slightly different than the reported ones. Notwithstanding, the conclusions should be similar as the reported ones.


## Repository structure

Common files:
- [`Dockerfile`](./Dockerfile): a recipe for the computational environment using Docker.
- [`requirements.txt`](./requirements.txt): file with the dependencies and versions used through all the code.

Code files and directories:

- [`functions`](./functions): contains several Python files defining functions used in the following scripts and notebooks.
- [`01_data-processing.py`](./01_data-processing.py): Python script implementing the data processing described in the paper (i.e., temporal aligment, Min-Max scaling, labelling and windowing). It reads the raw data from [`01_DATA/01_RAW`](./01_DATA/01_RAW) and stores the processed data in [`01_DATA/02_CLEAN`](./01_DATA/02_CLEAN) (labelled clean samples) and [`01_DATA/03_WINDOWED`](./01_DATA/03_WINDOWED) (windows). You can run the script by executing the following command in the terminal:
  ```bash
  python 01_data-processing.py
  ```
- [`02_incremental-loso.py`](./02_incremental-loso.py): Python script implementing the incremental cross-validation described in the paper. The performance metric obtained from the training process are stored in [`02_MODEL-REPORTS`](./02_MODEL-REPORTS). You can run the script with its default parameters by executing the following command in the terminal:

  > **Warning**: with the default parameters, the script will completely execute the incremental cross-validation described in the paper. This process will last some days, depending on hardware specifications.
  
  > **Note**: use the `--subject <number>` flag to execute the validation for only the indicated subject (1 to 23). These process will last a couple of hours, depending on hardware specifications. 

  ```bash
  python 02_incremental-loso.py
  ```
   
  > **Warning**: running the script will overwrite the results stored in `02_MODEL-REPORTS`. You can use the `--testing_script` flag to avoid overwriting the results.
   
  You can run the following command to obtain a description of the accepted parameters:
   
  ```bash
  python 02_incremental-loso.py -h
  ```

- [`03_performance-evaluation-analysis.ipynb`](./03_performance-evaluation-analysis.ipynb): Jupyter Notebook containing all the code used to process and analyse the performance metrics obtained from the trained models. Contains the results presented and discussed in the paper.

Directories and data files:

- [`01_DATA`](./01_DATA):
  - [`01_RAW`](./01_DATA/01_RAW): contains a raw accelerometer and gyroscope data collected from the participants of the study.
    - `sXX`: directory containing the raw data obtained from the subject XX. Each directory contains a `json` file for each activity execution sequence and for each device. Files have the following naming convention: `{subject_id}_{execution}_{sw|sp}.json`.
    - [`subjects.csv`](./01_DATA/01_RAW/subjects.csv): information (e.g., age, gender) regarding the participants of the study.
  - [`02_CLEAN`](./01_DATA/02_CLEAN): contains the processed accelerometer and gyroscope data, where each sample is labelled with an associated activity (i.e., SEATED, STANDING_UP, WALKING, TURNING and SITTING_DOWN). Contains a directory for each subject, and each directory has a `csv` file with the labelled data corresponding to one execution. The `csv` files follow this format: `{subject_id}_{execution}_{sw|sp}.csv`.
  - [`03_WINDOWED`](./01_DATA/03_WINDOWED): contains the windows generated from the clean data. The windows are labelled with the SEATED, STANDING_UP, WALKING, TURNING and SITTING_DOWN activities. Contains a directory for each subject with the following files:
    - `{subject_id}_{sp|sw}.npy`: _numpy_ file containing the smartphone or the smartwatch data windows.
    - `{subject_id}_{sp|sw}_gt.npy`: _numpy_ file containing the grountruth (i.e., associated label) for the windows in the data file.
- [`02_MODEL-REPORTS`](./02_MODEL-REPORTS): contains files with the performance metrics obtained from the trained models.
  - [`sp_models.csv`](./02_MODEL-REPORTS/sp_models.csv): performance metrics from models trained with smartphone data.
  - [`sw_models.csv`](./02_MODEL-REPORTS/sw_models.csv): performance metrics from models trained with smartwatch data.
  
    > **Note**: csv columns are the following ones:
    > - **test_subject**: subject use to evaluate the model
    > - **n**: number of subjects used for training.
    > - **i**: i<sup>th</sup> model trained for a certain **n**.
    > - **target**: to which element the **metric** belongs (e.g., model, certain activity, etc.).
    > - **metric**: to which metric belongs the **value** (e.g., accuracy, f1-score, etc.).
    > - **value**: value of the metric.
    
  - [`loso_groups.csv`](./02_MODEL-REPORTS/loso_groups.csv): indicates which subjects have been used to train and evaluate each model.
  
- [`03_RESULTS`](./03_RESULTS): contains the results obtained from the analyses in the [`03_performance-evaluation-analysis.ipynb`](./03_performance-evaluation-analysis.ipynb) notebook. Contains:
  - `01_{sp|sw}-ev.pdf`: plots with the overall accuracy evolution for the smartphone and smartwatch models. These correspond to the **Figure 2** of the paper.
  - `02_{activity}-ev.pdf`: plots with the activity-wise F1-score evolution for the smartphone and smartwatch models. These correspond to the **Figure 4** of the paper.
  - `03_sp-fails-by-subject.pdf`: plot with the number of times each evaluation subject obtained a F1-score == 0 in the _seated_, _standing up_ and _sitting down_ activities in the _sp_ models. The plot corresponds with the **Figure 5** of the paper.
  - `04_{sp|sw}-pairwise-table.html`: tables with the results of the statistical tests performed to find significant differences in the evolution of the overall performance.
  - `04_{sp|sw}-pairwise-tests.pdf`: graphical representations of the previous tables. These correspond to **Figure 3** of the paper.
  - `05_{overall|activities}-comparison-table.html`: tables with the results of the statistical tests performed to find significant differences in the evolution of the activity-wise performance. These tables correspond the the **Table 3** of the paper.


## License
The documents in this repository are licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

All contained code is licensed under the [Apache License 2.0](./LICENSE).

All data used in this repository is licensed under [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/).
