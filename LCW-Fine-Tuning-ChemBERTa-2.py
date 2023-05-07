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
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback, pipeline
import pandas as pd
import warnings
import numpy as np
import evaluate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars")
# define the input file
input_file: str = 'aqua.csv'
output_directory: str = './output'
# Define the maximum sequence length
max_length = 195


class MyData:
    def __init__(self, i_data):
        self.data = i_data

    def get_split(self, train_ratio=0.8, valid_ratio=0.1, seed=None):
        n = len(self.data)
        indices = np.arange(n)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        train_size = int(train_ratio * n)
        valid_size = int(valid_ratio * n)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        i_train_data = self.data.iloc[train_indices].reset_index(drop=True)
        i_valid_data = self.data.iloc[valid_indices].reset_index(drop=True)
        i_test_data = self.data.iloc[test_indices].reset_index(drop=True)
        return i_train_data, i_valid_data, i_test_data


class Input(Dataset):
    def __init__(self, i_data, i_tokenizer, i_max_length):
        self.data = i_data
        self.tokenizer = i_tokenizer
        self.max_length = i_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]["Standardized_SMILES"]
        inputs = self.tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.max_length)
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["labels"] = torch.tensor(self.data.iloc[idx]["median_WS"], dtype=torch.float).unsqueeze(0)
        return inputs


# Define a callback for printing validation loss
class PrintValidationLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if state is not None and hasattr(state, 'eval_loss'):
            print(f"Validation loss: {state.eval_loss:.4f}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Read in solubility data and split
def read_solubility(filename: str):
    my_data = pd.read_csv(filename)
    # Create an instance of the MyData class
    my_data = MyData(my_data)
    # Split your data into training, validation, and testing sets
    train_data, valid_data, test_data = my_data.get_split(seed=123)
    # pick out columns
    r_data = train_data[['Standardized_SMILES', 'median_WS']]
    r_valid = valid_data[['Standardized_SMILES', 'median_WS']]
    r_test = test_data[['Standardized_SMILES', 'median_WS']]
    return r_data, r_valid, r_test


# retrieve the device to move the model to
def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Using NV GPU.")
    # The mps device in torch does repeatedly lead to a RuntimeError: Placeholder storage has not been allocated
    # on MPS device!
    #elif torch.backends.mps.is_available():
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
        # max_length=195 and move the inputs also to the device
        inputs = tokenizer(i_smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=195).to(dev)
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
# run it for prepared smiles data, set a string set_type for output, device, and a flag to save results
def run_prediction(prep_smiles, set_type, dev, is_saved):
    out_r_mse, out_r2, out_mae, predictions, correlation, p_value = predict_smiles(prep_smiles, dev)
    display_results(set_type, out_r_mse, out_r2, out_mae, predictions, correlation, p_value)
    if is_saved:
        results_df = pd.DataFrame({"actual_WS": test["median_WS"], "predicted_WS": predictions})
        results_df.to_csv("testset_results.csv", index=False)


#
# main program
#

# AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
# Load a pretrained transformer model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers += 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# see if GPU and assign model (move model to the device)
device = get_device()
model.to(device)

# Read and prepare the dataset for training
data, valid, test = read_solubility(input_file)
train_dataset = Input(data, tokenizer, max_length)
validation_dataset = Input(valid, tokenizer, max_length)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_directory,
    optim="adamw_torch",  # switch optimizer to avoid warning
    num_train_epochs=100,  # Train the model for 100 epochs
    per_device_train_batch_size=128,  # Set the batch size to 128
    per_device_eval_batch_size=128,  # Set the evaluation batch size to 128
    logging_steps=10,  # Log training metrics every 100 steps
    eval_steps=10,  # Evaluate the model every 100 steps
    save_steps=10,  # Save the model every 100 steps
    seed=123,  # Set the random seed for reproducibility
    evaluation_strategy="steps",  # Evaluate the model every eval_steps steps
    load_best_model_at_end=True
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Add the callback to the trainer
trainer.add_callback(PrintValidationLossCallback())

metric = evaluate.load("accuracy")

# Train the model
trainer.train()

# Save the model
trainer.save_model("./output")

# Create a prediction pipeline
predictor = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Prepare new SMILES strings for prediction TRAINING-SET
run_prediction(data['Standardized_SMILES'], "TRAINING SET", device, False)

# Prepare new SMILES strings for prediction VALIDATION SET
run_prediction(valid['Standardized_SMILES'], "VALIDATION SET", device, False)

# Prepare new SMILES strings for prediction TEST SET
run_prediction(test['Standardized_SMILES'], "TEST SET", device, True)
