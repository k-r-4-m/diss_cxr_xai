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

## Datasets
The main training and evaluation dataset used in this project is VinDr-CXR. A slghtly modified version hosted on Kaggle is used, which can be found [here](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/overview).

Only the train dataset is used here since the test set does not have labels.

Textual radiology report evaluation is performed using the PadChest dataset, which can be requested [here](https://bimcv.cipf.es/bimcv-projects/padchest-gr/).

## File Organisation
After the VinDr-CXR dataset has been downloaded, the directory named "train" in the dataset should be renamed as "dicom" and placed in the root directory. The "train.csv" file in the dataset should be renamed to "train_original.csv" and placed in the root directory. 

The PadChest dataset only requires use of the .png files in the folder named "padchest_gr_dataset" and the .csv file named "master_table.csv". The folder with the images should be renamed to "images" and the .csv file should be renamed to "padchest_annotations.csv". The folder and the .csv file can then be placed in a folder in a root directory named "padchest".

This should lead to the following file organisation:
```
root
└──
  ├── dicom/
      └── [VINDR DICOM FILES GO HERE]
  ├── model_checkpoints/
      └── epoch_1...
      └── epoch_n
  ├── ensemble_checkpoints/
      └── epoch_1...
      └── epoch_n
  ├── padchest/
      └── images/
          └── [PADCHEST PNG FILES GO HERE]
      | padchest_annotations.csv   
  | ensemble.py
  | ensemble_eval.py
  | florence_after_aug.py
  | florence_classes_plot_after_voting.py
  | florence_eval_text.py
  | florence_eval.py
  | florence_preprocess_aug.py
  | florence_tools.py
  | florence_train.py
  | maira_eval.py
  | maira_eval_text.py
  | padchest_preprocessing.py
  | train_original.csv
```

## Data Preprocessing
Before training any of the models, the dataset needs to be preprocessed and augmented, which can be done so by running the vindr_preprocessing_aug.py which preprocesses VinDr-CXR into Florence format which all other models also use. padchest_preprocessing.py must also be run to preprocess PadChest-GR for textual radiology report evaluation.

## Training the Models
Running the code to train Florence-2 and the ensemble model can be done so by running florence_train.py and ensemble_train.py files respectively.

Please be aware that these models require **significant** resources and time to run. Training was performed on NVIDIA A100 GPUs and 300GB of system RAM. If you attempt to run these models on weak GPUs, you will most likely get out of memory errors. The code should be able to run using only CPU if a CUDA GPU is not available, but this will take substantially longer.

The model checkpoints used for the evaluation results in the dissertation report are stored in this repo.

## Evaluating the Models
Classification and localisation evaluation for Florence-2, the ensemble model, and MAIRA-2 can be performed by running the florence_eval.py, ensemble_eval.py, and maira_eval.py files respectively.

Textual radiology report evaluation for Florence-2 and MAIRA-2 can be performed by running the florence_eval_text.py and maira_eval_text.py files respectively.

