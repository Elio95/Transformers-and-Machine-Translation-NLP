import os

from dataModels.dataModel import dataModel
from dataModels.ChnSentiCorp import ChnSentiCorp
from dataModels.imdb import IMDB


class MixedDatasets(dataModel):
    def __init__(self):
        super().__init__()
        self.dataset_dir = os.path.join(".", "datasets")

    def load_data(self):
        chinSentiCorp = ChnSentiCorp()
        imdb = IMDB()

        # training data
        self.chin_train_df = chinSentiCorp.chin_train_df + imdb.chin_train_df
        self.eng_train_df = chinSentiCorp.eng_train_df + imdb.eng_train_df

        # test data
        self.chin_test_df = chinSentiCorp.chin_test_df + imdb.chin_test_df
        self.eng_test_df = chinSentiCorp.eng_test_df + imdb.eng_test_df

        # dev data
        self.chin_dev_df = chinSentiCorp.chin_dev_df + imdb.chin_dev_df
        self.eng_dev_df = chinSentiCorp.eng_dev_df + imdb.eng_dev_df
