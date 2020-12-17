# COMP550_FinalProject

#### Data Models 
  ###### Preprocessing : `translator.py`
For the data, we made sur that the file where split into 3 files : train, dev and test with the file format tsv. When we have the 3 files in format tsv, we can proceed to translate them into the other language. The translated files are denoted by `<language_symbol>_<original file name>.py` 

  ###### New data type : `dataModel.py`
We decided to create a super class called dataModel. This has all the basic functions: how to get the chinese data and the english data. The sub classes only have to overload `load_data` the function that loads the data into to the correct variables from the right files. 


#### Models 
We decided to create a new data type : `Model.py`. This ensures that all the models that we create use the same hyperparameters.  

Example on how to run it on [Google colab](https://colab.research.google.com/drive/1H0sWs2pjROFxXRokv0oG0AWCnFvYeRRq#scrollTo=Qhdl0oS3LW3k)
