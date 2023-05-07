#!/usr/bin/env python
# coding: utf-8
#
# Released under MIT License
#
# Copyright (c) 2023 Andrew SID Lang, Oral Roberts University, U.S.A.
#
# Copyright (c) 2023 Jan HR Woerner, Oral Roberts University, U.S.A.
#
# Copyright (c) 2023 Wei-Khiong (Wyatt) Chong, Advent Polytech Co., Ltd, Taiwan.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from transformers import pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
output_directory = "./output"
# max length of SMILES over both sets
max_length = 195

class Input(Dataset):
    def __init__(self, i_data, i_tokenizer, i_max_length):
        self.data = i_data
        self.tokenizer = i_tokenizer
        self.max_length = i_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        i_smiles = self.data.iloc[idx]["Standardized_SMILES"]
        i_inputs = self.tokenizer(i_smiles, return_tensors="pt", padding='max_length', truncation=True,
                                  max_length=self.max_length)
        i_inputs["input_ids"] = i_inputs["input_ids"].squeeze(0)
        i_inputs["attention_mask"] = i_inputs["attention_mask"].squeeze(0)
        if "token_type_ids" in i_inputs:
            i_inputs["token_type_ids"] = i_inputs["token_type_ids"].squeeze(0)
        i_inputs["labels"] = torch.tensor(self.data.iloc[idx]["median_WS"], dtype=torch.float).unsqueeze(0)
        return i_inputs


# retrieve the device to move the model to
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Using NV GPU.")
    # The mps device in torch does repeatedly lead to a RuntimeError: Placeholder storage has not been allocated
    # on MPS device!
    #elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #    dev = torch.device("mps")
    #    print("Using M1 GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        dev = torch.device("cpu")
    return dev


# Predict properties for new SMILES strings
def predict_smiles(u_smiles, dev):
    preds = []
    for i_smiles in u_smiles:
        inputs = tokenizer(i_smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=195).to(dev)
        # max_length=195
        with torch.no_grad():
            outputs = model(**inputs)
    pred_property = outputs.logits.squeeze().item()
    preds.append(pred_property)
    r_mse = mean_squared_error(data["median_WS"], preds, squared=False)
    r2 = r2_score(data["median_WS"], preds)
    mae = mean_absolute_error(data["median_WS"], preds)
    correlation, p_value = spearmanr(data["median_WS"], preds)
    return r_mse, r2, mae, preds, correlation, p_value


# display the results
def display_results(dataset_type, in_r_mse, in_r2, in_mae, preds, correlation, p_val):
    print(dataset_type)
    print("N:", len(data["median_WS"]))
    print("R2:", in_r2)
    print("Root Mean Square Error:", in_r_mse)
    print("Mean Absolute Error:", in_mae)
    print("Spearman correlation:", correlation)
    print("p-value:", p_val)

    plt.scatter(data["median_WS"], preds)
    plt.xlabel("train['median_WS']")
    plt.ylabel("predictions")
    plt.title("Scatter Plot of " + dataset_type + " ['median_WS'] vs Predictions")
    plt.show()


# assume test and predictions are two arrays of the same length
# run it for train smiles data
def run_prediction(prep_smiles, set_type, dev, is_saved):
    out_r_mse, out_r2, out_mae, predictions, correlation, p_value = predict_smiles(prep_smiles, dev)

    display_results(set_type, out_r_mse, out_r2, out_mae, predictions, correlation, p_value)
    if is_saved:
        results_df = pd.DataFrame({"actual_WS": test["median_WS"], "predicted_WS": predictions})
        results_df.to_csv("testset_results.csv", index=False)

#
# LCW Using ChemBERTa-2 For Property Prediction main program
#

# Read in solubility data
train_data = pd.read_csv('aqua_train.csv')
test_data = pd.read_csv('aqua_test.csv')

# pick out columns
data = train_data[['Standardized_SMILES', 'median_WS']]
test = test_data[['Standardized_SMILES', 'median_WS']]

# Load a pretrained transformer model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers += 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# move model to the device
device = get_device()
model.to(device)

# Prepare the dataset for training
train_dataset = Input(data, tokenizer, max_length)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_directory,
    num_train_epochs=100,  # Number of training epochs
    per_device_train_batch_size=86,  # Batch size
    logging_steps=100,  # Log training metrics every 100 steps
    optim="adamw_torch",  # switch optimizer to avoid warning
    seed=123,  # Set a random seed for reproducibility
)

# Train the model
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, )
trainer.train()
trainer.save_model("./output")  # save model to output folder

# Create a prediction pipeline

predictor = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Prepare new SMILES strings for prediction and run the model for test data
test_smiles = test['Standardized_SMILES']
run_prediction(test_smiles, "TEST SET", device, False)

# Prepare new SMILES strings for prediction and run the model for training data
train_smiles = data['Standardized_SMILES']
run_prediction(train_smiles, "TEST SET", device, False)



