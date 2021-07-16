# ERSW-project
### This is the code repository for the manuscript: “An interpretable early dynamic sequential predictor for sepsis-induced coagulopathy progression in the real-world using AI "

#### The repository provides the code of our research reproduction as follows:
1. Installing necessary Python dependencies
2. Generate or acquire the CSV files which are used for analysis by...
  - Running the **.py** file in the **data_preprocessing** folder to complete the data segmentation, sampling time window division and disease status annotation.
  - Running the **.py** file in the **model_development** folder to develop LR, SVM, LightGBM, XGBoost, LSTM, RNN, RNN-Decay and ODE-RNN models.
  - Running the **.py** file in the **model_evaluation** folder to plot ROC curves, PR curves and model interpretation.

BIDMC set data were obtained from the MIMICIII database, which can be downloaded from https://mimic.mit.edu/iii . There are two options acquire CSVs from a database with MIMIC-III: (a) regenerate the CSVs from the original MIMIC-III database, or (b) download pre-generated CSVs from PhysioNetWorks. XJTUMC set data were obtained from the Biobank of First Affiliated Hospital of Xi’an Jiaotong University, which is a restricted-access resource and is only available by submitting a request to the author and the institution.

