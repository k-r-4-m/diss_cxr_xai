# Master's Dissertation: Trust and Transparency in Healthcare Using Explainable AI 

## Install dependencies
The various models used in this project require different dependencies.

To install dependencies for Florence-2 and the ensemble models, execute the following commands:
```
conda create -n florence python=3.13.5 -y
conda activate florence
pip install -r florence_requirements.txt
```

To install dependencies for MAIRA-2, execute the following commands:
```
conda create -n maira_2 python=3.13.5 -y
conda activate maira_2
pip install -r maira_2_requirements.txt
```

The use of MAIRA-2 requires a HuggingFace token. Please place this token into the token_file.yaml file.

## Dataset
The dataset used in this project is VinDr-CXR. A slghtly modified version hosted on Kaggle is used, which can be found [here](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview).

Only the train dataset is used here since the test set does not have labels.

## File Organisation
After the dataset has been downloaded, the directory named "train" in the dataset should be renamed as "dicom" and placed in the root directory. The "train.csv" file in the dataset should be renamed to "train_original.csv" and placed in the root directory. 

This should lead to the following file organisation:
```
root
└──
  ├── dicom/
      └── [DICOM FILES GO HERE]
  ├── model_checkpoints/
      └── epoch_1...
      └── epoch_n
  ├── model_checkpoints_no_defs/
      └── epoch_1...
      └── epoch_n
  ├── ensemble_checkpoints/
      └── epoch_1...
      └── epoch_n
  ├── XYZ_checkpoints/
      └── epoch_1
      └── epoch_n
  | ensemble.py
  | ensemble_eval.py
  | florence_after_aug.py
  | florence_eval.py
  | florence_eval_phrase_grounding.py
  | florence_phrase_grounding_convert.py
  | florence_preprocess_aug.py
  | florence_tools.py
  | florence_train.py
  | train_original.csv
```

## Data Preprocessing
Before training any of the models, the dataset needs to be preprocessed and augmented, which can be done so by running the florence_preprocessing_aug.py file.

## Training the Models
Running the code to train Florence-2, the ensemble model, and XYZ can be done so by running florence_train.py, ensemble_train.py, and XYZ_train.py files respectively.

Please be aware that these models require **significant** resources and time to run. Training was performed on NVIDIA A100 GPUs and 300GB of system RAM. If you attempt to run these models on weak GPUs, you will most likely get out of memory errors. The code should be able to run using only CPU if a CUDA GPU is not available, but this will take substantially longer.

The model checkpoints used for the evaluation results in the dissertation report are stored in this repo.

## Evaluating the Models
Model evaluation can be performed by running the florence_eval.py, ensemble_eval.py, and XYZ_eval.py files respectively.

