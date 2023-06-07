# About
This project attempts to find the correlation between the controversialness and human values of Reddit's r/ChangeMyView comments. It finds the human values using a trained DeBERTa v3 model.

# Usage
This project has been tested using Python 3.9. The results were gathered using a GPU. 

The results can be gathered by running `main.py`. The `best_model.pt` (a DeBERTa v3 model with a learning rate of 5e-5 and a patience of 11) is used to find the human values. This model can be added in the `data` folder.

## Results analysis
To perform an analysis of the results in `results/complete_results.csv` (not included in this repo due to size), run `perform_results_analysis.py`. This will compute the Jaccard's distance between the values, generate a wordcloud of controversial and non controversial comments, and fit a logistic regression.

# Directory Structure

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
│   │   ├── complete_results
│   │   └── controversy_pred_results
├── get_values.py
├── get_results.py
├── read_data.py
├── requirements.txt
├── run_model.py
└── main.py
```

The `\data` folder contains the .csv files containing the Reddit comments posted in various years (needs to be added by the user). The data is loaded as a tensor for PyTorch in `read_data.py`.

The `\models` folder contains a .pt PyTorch model that is used to detect the human values (needs to be added by the user).

The `\results` folder contains a list `complete_results` which has information for each comment (e.g. the author, time, text), the comment's controversialness score (taken from Reddit) and the comment's human values (determined by the model). The `controversy_pred_results` contains only a comment's controversialness score and the comment's human values. These are the results that the read_data.py produces.


