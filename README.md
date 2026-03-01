LCW-Fine-Tuning-ChemBERTa-2

This python project is about fine tuning the pretrained DeepChem ChemBERTa 77M MTR transformer model to predict aqueous solubility values expressed as median_WS directly from molecular SMILES representations. Training and test datasets are loaded from CSV files and the SMILES strings are tokenized using the associated ChemBERTa tokenizer. A custom PyTorch Dataset class formats the tokenized molecular representations together with their corresponding experimental solubility values as continuous regression targets. The model is configured with a single output neuron to perform regression and is trained for one hundred epochs using the AdamW optimization algorithm within the Hugging Face Trainer framework. Hardware acceleration is automatically enabled when a compatible GPU is available. After training, the optimized model parameters are saved and the model is used to generate predictions for both training and test datasets.

Model performance is evaluated using multiple regression metrics including the coefficient of determination R squared, root mean square error, mean absolute error, and Spearman rank correlation. In addition to numerical evaluation, a scatter plot is generated to visualize the relationship between experimental and predicted solubility values. The example training set plot shows a strong linear association between predicted and observed values, indicating that the fine tuned ChemBERTa model effectively captures relevant structure property relationships encoded in the molecular representations. The concentration of data points along the diagonal trend reflects good agreement between model predictions and experimental measurements, while deviations from this trend correspond to residual prediction error.

Output:

![image](https://github.com/user-attachments/assets/648cbdfd-8317-4000-ada7-7a9fd1326209)

![image](https://github.com/user-attachments/assets/73428483-e87c-4998-a8fa-6b7fc551c592)

![image](https://github.com/user-attachments/assets/31066c53-ac6f-45e1-8136-bae0dd2b1d50)

![image](https://github.com/user-attachments/assets/601e96dc-e37b-4122-aa7f-54543fab8cf2)

![image](https://github.com/user-attachments/assets/c636af2f-5a98-477c-97ff-0c3e9df2637d)
