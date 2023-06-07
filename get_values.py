import numpy as np
import torch

def get_values(loaded_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

    model = torch.load("models/best_model.pt")

    model.eval()

    all_preds = []

    counter = 0

    with torch.no_grad():
        for batch in loaded_data:
            input_ids_batch, input_mask_batch = batch

            print(input_ids_batch[0].shape)

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

    return all_preds
