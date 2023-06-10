# About
This project attempts to find the correlation between the controversialness and human values of Reddit's r/ChangeMyView comments. It finds the human values using [this](https://github.com/sandra-bee/human-value-detection) trained DeBERTa v3 model.

# Usage
This project has been tested using Python 3.9. The results were gathered using a GPU. 

## Pre-requisites
Install the requirements in `requirements.txt` using `pip install -r requirements.txt` in your terminal (we recommend 
setting up a virtual environment). 

The `best_model.pt` (a DeBERTa v3 model with a learning rate of 5e-5 and a patience of 11) is used to find the human values and can be downloaded from [here](https://drive.google.com/file/d/1wbznu605vpJtr3YErOkl27cynj95XMpX/view?usp=drive_link) with a RuG gmail account. This model should be added in the `models/` folder. 

For the dataset, download `threads.jsonl.bz2` from [here](https://zenodo.org/record/3778298) and place the extracted `threads.jsonl` file in the main directory. Datafiles of lists of comments per year is obtained by running the script of `preprocess_data.py` on the r/ChangeMyView dataset and should be added in the `data/` folder.

To run results analysis, download the complete results from [here](https://drive.google.com/file/d/1W-LQODa3AQUf-WvZTkrt70zhjeGcojXo/view?usp=drive_link) with your RuG gmail account and place it in `results/`.

## Launch
The results can be gathered by running `main.py`. 


## Results analysis
To perform an analysis of the results in `results/complete_results.csv` (not included in this repo due to size), run `perform_results_analysis.py`. This will compute the Jaccard's distance between the values, generate a wordcloud of controversial and non controversial comments, and fit a logistic regression. 
To obtain barplots of the instances of controversiality present in the dataset per value as well as percentages of these, run `plot_results_final.py`. This will generate barplots which will be stored in `plots/`.
# Directory Structure

This project is structured as follows:
```bash
controversialness_human_values_correlation_analysis
│   ├── data
│   │   ├── 2016.csv
│   │   └── ...
│   ├──models
│   │   ├── best_model.pt
│   ├──plots
│   │   ├── plot1_400_filtered_final.png
│   │   └── ...
│   ├──results
│   │   ├── complete_results
│   │   └── controversy_pred_results
├── main.py
├── perform_results_analysis.py
├── plot_results_final.py
├── preprocess_data.py
├── read_data.py
├── read_results.py
├── requirements.txt
└── train.sh
```

The `\data` folder contains the .csv files containing the Reddit comments posted in various years (needs to be added by the user). The data is loaded as a tensor for PyTorch in `read_data.py`.

The `\models` folder contains a .pt PyTorch model that is used to detect the human values (needs to be added by the user).

The `\results` folder contains a list `complete_results` which has information for each comment (e.g. the author, time, text), the comment's controversialness score (taken from Reddit) and the comment's human values (determined by the model). The `controversy_pred_results` contains only a comment's controversialness score and the comment's human values. These are the results that the read_data.py produces.

`train.sh` can be used to run model testing on Habrok.
