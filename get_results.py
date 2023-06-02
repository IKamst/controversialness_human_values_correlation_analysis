import csv

import numpy as np


def get_results_lists(data, predictions):
    controversy_pred_list = []
    for index, prediction in enumerate(predictions):
        controversy_pred_list.append([data[index][2], prediction])
        data[index].append(prediction)
    with open('results/controversy_pred_results', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(controversy_pred_list)
    with open('results/complete_results', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        write.writerows(data)

def get_results(data, predictions):
    get_results_lists(data, predictions)
    return