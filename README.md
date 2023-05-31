# About
This project attempts to find the correlation between  solve the Human Value Detection 2023 task  [[1]](#1) using the DeBERTa v3 model.

# Usage
This project has been tested using Python 3.10 and is trained on a GPU. 

This project expects the following files in a `data` folder (not included in this repo, can be downloaded from [[1]](#1)):
* arguments-training.tsv
* arguments-validation.tsv
* arguments-test.tsv
* labels-training.tsv
* labels-validation.tsv
* labels-test.tsv

Install the requirements in `requirements.txt`, then run `main.py <train|test>` to launch. The parameter `train` will launch either grid search (if `GRID_SEARCH=True` in main) or will perform training with a set of optimal
hyperparameter values chosen. The parameter `test` will perform testing using the `best_model.pt` that has been saved in folder `models\` after training. You can set `MAKE_PLOTS=True` in main to visualise the training and validation loss obtained on the best model.

## Data Augmentation
To run data augmentation on the files in `\data`, run the file `DataAugmentation.py`. This will generate the following augmented training data files:
* augmented-arguments-training.tsv
* augmented-labels-training.tsv

You can then train the model using the augmented training data by setting the flag `USE_DATA_AUG=True` in main.

# Directory Structure

[TODO update this]

This project is structured as follows:
```bash
controversialness_human_values_correlation_analysis
│   ├── data
│   │   ├── 2016.csv
│   │   └── ...
│   ├──models
│   │   ├── best_model.pt
│   │   └── ...
│   ├──plots
│   │   ├── ...
│   │   └── ...
├── load_data.py
├── run_model.py
└── main.py
```

The `\data` folder contains the .csv files containing the arguments and their respective labels (see 'Usage'). The data is loaded as a tensor for PyTorch in `data_preprocessing.py` and augmented in `LoadData.py` and `DataAugmentation.py`.

The `\models` folder contains the .pt PyTorch models that are generated after finetuning DeBERTa to solve this task. The models saved are the best ones (i.e. the model configuration where F1 score for the validation set was maximal.) The models are made in `model_making.py`.

The `\plots` folder contains the loss curves generated during training and validation. These plots are generated in `visualisation.py`.

The `\result_metrics` folder contains lists of F1 scores and loss values that are generated to make plots. E.g. the list of training losses per epoch are stored such that they can be plotted later. These lists are stored in `store_metrics.py`.


# References

<a id="1">[1]</a>  https://touche.webis.de/semeval23/touche23-web/index.html
