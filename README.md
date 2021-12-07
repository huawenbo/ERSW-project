# ERSW-project

#### This is the code repository for the article: [*An interpretable early dynamic sequential predictor for sepsis-induced coagulopathy progression in the real-world using AI*](https://www.frontiersin.org/articles/10.3389/fmed.2021.775047/full), which is published in Frontiers in Medicine.  

### 1. Here are some results of the article.

### 2. This is the code structure of ERSW-project-main.

- Code structure

  > ERSW-project-master
  > > data  
  > > > BIDMC set and XJTUMC set
  > 
  > > lib  
  > > > deep_learning_model.py  
  > > > machine_learning_model.py  
  > > > figure_plotting.py  
  > > > shap_plotting.py
  > 
  > > utils
  > > > data_dividation.py  
  > > > get_sample.py  
  > > > merge_annotate.py  
  > > > pre_annotation.py  
  >  
  > > <text>main.py</text>  

### 3. The repository provides the code of our research reproduction as follows:

- Install necessary Python dependencies, such as "torch", 'shap', and so on.
- Acquire or generate the necessary Dataset which are used for analysis by the following ways.
    - **BIDMC** dataset were obtained from the MIMIC-III database, which can be downloaded from *https://mimic.mit.edu/iii*.
    - **XJTUMC** dataset were obtained from the Biobank of First Affiliated Hospital of Xi’an Jiaotong University, which is a restricted-access resource and is only available by submitting a request to the author and the institution. You can send your request to the email: *hwb0856@stu.xjtu.edu.cn*
- Run *<text>main.py</text>* to deal with the data and develop the model for predicting the coagulopathy.
