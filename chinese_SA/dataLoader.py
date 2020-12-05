import codecs
import csv
import os

import pandas as pd
import tensorflow as tf

URL = "https://paddlehub-dataset.bj.bcebos.com/chnsenticorp.tar.gz"

class DataLoader:
    def __init__(self):
        self.dataset_dir = os.path.join(".", "datasets/chnsenticorp")
        file_path = tf.keras.utils.get_file(
            fname="chnsenticorp.tar.gz",
            origin=URL,
            extract=True,
            cache_dir=".",
        )
        self.train_file, self.train_df, self.train_num = None
        self.dev_file, self.dev_df, self.dev_num = None
        self.test_file, self.test_df, self.test_num = None

        self.load_data()

    def load_data(self):
        """
        Update the variables with the input
        :return:
        """
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.train_df = self.get_df_from_file(self.train_file)
        self.train_num = len(self.train_df)

        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.dev_df = self.get_df_from_file(self.test_file)
        self.dev_num = len(self.dev_df)

        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.test_df = self.get_df_from_file(self.test_file)
        self.test_df["labels"] = self.test_df["labels"].apply(lambda x: x[0])
        self.test_num = len(self.test_df)

    def get_train_df(self):
        """
        Return the data frame containing the training data
        Contains columns text: str, labels: list<int>
        :return: train_df
        """
        return self.train_df

    def get_dev_df(self):
        """
        Return the data frame containing the development data
        Contains columns text: str, labels: list<int>
        :return: dev_df
        """
        return self.dev_df

    def get_test_df(self):
        """
        Return the data frame containing the testing data
        Contains columns text: str, labels: int
        :return: test_df
        """
        return self.test_df

    def get_labels(self):
        """
        :return: list of labels
        """
        return ["0", "1"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def get_df_from_file(self, input_file):
        """
        Convert the file to a pandas.DataFrame
        :param input_file: file containing the data
        :return: file converted
        """
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t")
            data = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                data.append([line[1], line[0]])
                seq_id += 1
            df = pd.DataFrame(data)
            df.columns = ["text", "labels"]
            df["labels"] = df["labels"].apply(lambda x: list(map(int, x)))
            return df

    def print_info(self):
        """
        Print the heads of all sets Train, Dev and testing with their sizes
        :return: None
        """
        print("training data:")
        print(self.get_train_df().head())
        print("\n")

        print("dev data:")
        print(self.get_dev_df().head())
        print("\n")

        print("test data:")
        print(self.get_test_df().head())
        print("\n")

        print("Train number:{}, Dev number:{}, Test number:{}".format(self.train_num, self.dev_num, self.test_num))
