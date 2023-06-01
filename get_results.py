###
# get_results.py
# LTP Task 2
#
# Get results data made up of the controversialness score and human values (and other information on the comments).
###

# Import package:
import csv

# Make both a data file containing the controversialness and the human values, and a data file consisting of this data,
# plus other information on the data (information that was in the input data, like the time, author and text itself):
def get_results_lists(data, predictions):
    controversy_pred_list = []
    for index, prediction in enumerate(predictions):
        controversy_pred_list.append([data[index][2], prediction])
        data[index].append(prediction)
    with open('results/controversy_pred_results', 'w') as f:
        write = csv.writer(f)
        write.writerows(controversy_pred_list)
    with open('results/complete_results', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerows(data)

# Get results data made up of the controversialness score and human values (and other information on the comments).
def get_results(data, predictions):
    get_results_lists(data, predictions)
    return