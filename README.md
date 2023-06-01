[TODO update]

# About
This project attempts to find the correlation between the controversialness and human values of Reddit's r/ChangeMyView comments. It finds the human values using a trained DeBERTa v3 model.

# Usage
This project has been tested using Python 3.9. The results were gathered using a GPU. 

The results can be gathered by running `main.py`. The `best_model.pt` (a DeBERTa v3 model with a learning rate of 5e-5 and a patience of 11) is used to find the human values. 

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
│   ├──results
│   │   ├── complete_results.pt
│   │   ├── controversy_pred_results.pt
│   ├──plots
│   │   ├── ...
│   │   └── ...
├── get_values.py
├── get_results.py
├── read_data.py
├── requirements.txt
├── run_model.py
└── main.py
```

The `\data` folder contains the .csv files containing the Reddit comments posted in various years. The data is loaded as a tensor for PyTorch in `read_data.py`.

The `\models` folder contains the .pt PyTorch models that are generated after finetuning DeBERTa during training. The model saved is the best ones (i.e. the model configuration where F1 score for the validation set was maximal for the training set).

The `\plots` folder contains..........

The `\results` folder contains a list `complete_results` which has information for each comment (e.g. the author, time, text), the comment's controversialness score (taken from Reddit) and the comment's human values (determined by the model). The `controversy_pred_results.pt` contains only a comment's controversialness score and the comment's human values.


