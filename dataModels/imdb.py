import os
from dataModels.dataModel import dataModel

class IMDB(dataModel):
    def __init__(self):
        super().__init__()
        self.set_dir(os.path.join(".", "datasets/IMDB"))

        self.load_data()

    def load_data(self):
        """
        Update the variables with the input
        :return:
        """

        print("load chinese training data")
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.eng_train_df = self.get_df_from_file(self.train_file)
        self.chin_train_df = self.get_df_from_file(os.path.join(self.dataset_dir, "zh_train.tsv"))
        self.train_num = len(self.chin_train_df)

        print("load dev data")
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.eng_dev_df = self.get_df_from_file(self.dev_file)
        self.chin_dev_df = self.get_df_from_file(os.path.join(self.dataset_dir, "zh_dev.tsv"))
        self.dev_num = len(self.chin_dev_df)

        print("load test data")
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.eng_test_df = self.get_df_from_file(self.test_file)
        self.eng_test_df["labels"] = self.eng_test_df["labels"].apply(lambda x: x[0])
        self.chin_test_df = self.get_df_from_file(os.path.join(self.dataset_dir, "zh_test.tsv"))
        self.chin_test_df["labels"] = self.chin_test_df["labels"].apply(lambda x: x[0])
        self.test_num = len(self.chin_dev_df)

        print("loading Chinese data done")