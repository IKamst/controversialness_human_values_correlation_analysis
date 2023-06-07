from read_data import read_data
from get_values import get_values
from get_results import get_results

import torch

if __name__ == '__main__':
    torch.set_default_device('cuda')  # TODO currently using GPU
    data, data_loader = read_data()
    predictions = get_values(data_loader)
    get_results(data, predictions)