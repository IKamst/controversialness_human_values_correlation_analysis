###
# read_data.py
# LTP Task 2
#
# Read Reddit comment data from a .csv file and store the data in a Pytorch tensor.
###

# Import packages:
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

# Read the data from a .csv file:
def get_data_from_csv():
    path = "data/data_2016.csv"
    with open(path, newline='', encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile, delimiter=";"))

    # TODO currently only get a subset of the data to calculate results on:
    return data[:200]

# Get the maximum token length of the comments by looking at the number of words:
def retrieve_max_token_len(sentence_list):
    max_sentence_length = 0
    for sentence in sentence_list:
        if len(str(sentence).split()) > max_sentence_length:
            max_sentence_length = len(str(sentence).split())
    return max_sentence_length

# Preprocess the data by padding (using the maximum sentence length), tokenizing and making a DataLoader of tensors:
def preprocess_data(data_texts):
    max_sentence_length = retrieve_max_token_len(data_texts)

    # Encoding:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Set is_split_into_words=True because the concatenated_train_x_set is a list of strings
    encoding = tokenizer.batch_encode_plus(data_texts,
                                           add_special_tokens=True,
                                           max_length=max_sentence_length,
                                           return_token_type_ids=False,
                                           padding='max_length',
                                           truncation=True,
                                           return_attention_mask=True,
                                           is_split_into_words=True)

    input_id = torch.tensor(encoding['input_ids'])
    attention_mask = torch.tensor(encoding['attention_mask'])

    # Put the x and y together:
    tensor_dataset = TensorDataset(input_id, attention_mask)

    return DataLoader(tensor_dataset, batch_size=16)

# Get the data from the .csv and preprocess it:
def read_data():
    data = get_data_from_csv()
    data_texts = [[row[1]] for row in data]

    data_loader = preprocess_data(data_texts)

    return data, data_loader