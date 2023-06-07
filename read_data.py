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
import pandas as pd

torch.set_default_device('cuda')

# Read the data from a .csv file: 2433815 2433825
def get_data_from_csv():
    return pd.read_csv("data/data_2016.csv", sep=";", chunksize=1000)

# Get the maximum token length of the comments by looking at the number of words:
def retrieve_max_token_len(sentence_list):
    max_sentence_length = 0
    for sentence in sentence_list:
        if len(str(sentence).split()) > max_sentence_length:
            max_sentence_length = len(str(sentence).split())
    return max_sentence_length

# Preprocess the data by padding (using the maximum sentence length), tokenizing and making a DataLoader of tensors:
def preprocess_data(chunk):
    data_texts = chunk.iloc[:,1].tolist()
    # print(type(data_texts))
    # data_texts = list(filter(lambda item: item != "[deleted]" and len(item) > 0, data_texts))
    data_texts = [[str(item)] for item in data_texts]
    # [print(f"Not good: {item}") for item in data_texts if len(item) == 0 or type(item) != str]

    # print(len(data_texts))
    # print(data_texts[:10])

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





# Import packages:
import numpy as np
import torch

# Get the human values from text data using a trained DeBERTa model.
def get_values(loaded_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO fix if using GPU!
    
    model = torch.load("models/best_model.pt",
                       map_location=torch.device(device))  # TODO remove 'map_location' if using GPU!

    model.eval()

    all_preds = []

    counter = 0

    with torch.no_grad():
        print(len(loaded_data))
        for batch in loaded_data:
            input_ids_batch, input_mask_batch = batch

            # Forward pass
            eval_output = model(input_ids=input_ids_batch, token_type_ids=None,
                                attention_mask=input_mask_batch)

            # Get predictions by applying sigmoid + thresholding:
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(eval_output.logits.cpu())
            preds = np.zeros(probs.shape)
            preds[np.where(probs >= 0.5)] = 1

            all_preds.extend(preds)

            counter = counter + 1
            print(counter)

    return all_preds



# Import package:
import csv

# Make both a data file containing the controversialness and the human values, and a data file consisting of this data,
# plus other information on the data (information that was in the input data, like the time, author and text itself):
def get_results_lists(data, predictions):
    # print("get_results_lists")
    # print(data[:10])
    # print(predictions[:10])

    # pd.DataFrame(new_data, index=[0]).to_csv(filename, sep=';', mode='a', header=False, index=False)
    # pd.DataFrame(new_data, index=[0]).to_csv(filename, sep=';', mode='a', header=False, index=False)

    data_df = pd.DataFrame(data)
    predictions_df = [[prediction] for prediction in predictions]
    # print(data_df.shape)
    # print(len(predictions_df))
    data_df.insert(5, "prediction", predictions_df)
    # print("here")
    # print(data_df)
    data_df.to_csv("results/complete_results_2016.csv", sep=';', mode='a', header=False, index=False)

    # controversy_pred_list = []
    # for index, prediction in enumerate(predictions):
    #     controversy_pred_list.append([data[index][2], prediction])
    #     data[index].append(prediction)
    # with open('results/controversy_pred_results', 'a') as f:
    #     write = csv.writer(f)
    #     write.writerows(controversy_pred_list)
    # with open('results/complete_results', 'a', encoding='utf-8') as f:
    #     write = csv.writer(f)
    #     write.writerows(data)

# Get results data made up of the controversialness score and human values (and other information on the comments).
def get_results(data, predictions):
    get_results_lists(data, predictions)
    return


# Get the data from the .csv and preprocess it:
def read_data():
    data = get_data_from_csv()
    # print(data.shape)
    for chunk in data:
        data_loader = preprocess_data(chunk)
        predictions = get_values(data_loader)
        
        # print(len(data))
        # print(len(predictions))
        # print(predictions)

        get_results(chunk, predictions)


