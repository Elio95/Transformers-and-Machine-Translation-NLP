import os
from dataModels.dataModel import dataModel

class ChnSentiCorp(dataModel):
    def __init__(self):
        super().__init__()
        self.set_dir(os.path.join(".", "datasets/chnsenticorp"))

        self.load_data()

    def load_data(self):
        """
        Update the variables with the input
        :return:
        """

        print("load chinese training data")
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.chin_train_df = self.get_df_from_file(self.train_file)
        self.eng_train_df = self.get_df_from_file(os.path.join(self.dataset_dir, "en_train.tsv"))
        self.train_num = len(self.chin_train_df)

        print("load dev data")
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.chin_dev_df = self.get_df_from_file(self.dev_file)
        self.eng_dev_df = self.get_df_from_file(os.path.join(self.dataset_dir, "en_dev.tsv"))
        self.dev_num = len(self.chin_dev_df)

        print("load test data")
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.chin_test_df = self.get_df_from_file(self.test_file)
        self.chin_test_df["labels"] = self.chin_test_df["labels"].apply(lambda x: x[0])
        self.eng_test_df = self.get_df_from_file(os.path.join(self.dataset_dir, "en_test.tsv"))
        self.eng_test_df["labels"] = self.eng_test_df["labels"].apply(lambda x: x[0])
        self.test_num = len(self.chin_dev_df)

        print("loading Chinese data done")
