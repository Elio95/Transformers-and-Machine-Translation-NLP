import codecs
import csv
import os

import pandas as pd

from abc import ABC, abstractmethod

class dataModel(ABC):
    def __init__(self):
        self.dataset_dir = None

        self.train_file, self.chin_train_df, self.eng_train_df, self.train_num = None, None, None, None
        self.dev_file, self.chin_dev_df, self.eng_dev_df, self.dev_num = None, None, None, None
        self.test_file, self.chin_test_df, self.eng_test_df, self.test_num = None, None, None, None

    def get_chin_train_df(self):
        """
        Return the data frame containing the training data
        Contains columns text: str, labels: list<int>
        :return: chin_train_df
        """
        return self.chin_train_df

    def get_eng_train_df(self):
        """
        Return the data frame containing the training data
        Contains columns text: str, labels: list<int>
        :return: eng_train_df
        """
        return self.eng_train_df

    def get_chin_dev_df(self):
        """
        Return the data frame containing the development data
        Contains columns text: str, labels: list<int>
        :return: chin_dev_df
        """
        return self.chin_dev_df

    def get_eng_dev_df(self):
        """
        Return the data frame containing the development data
        Contains columns text: str, labels: list<int>
        :return: dev_df
        """
        return self.eng_dev_df

    def get_chin_test_df(self):
        """
        Return the data frame containing the testing data
        Contains columns text: str, labels: int
        :return: chin_test_df
        """
        return self.chin_test_df

    def get_eng_test_df(self):
        """
        Return the data frame containing the english testing data
        Contains columns text: str, labels: int
        :return: eng_test_df
        """
        return self.eng_test_df

    @abstractmethod
    def load_data(self):
        pass

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
        with open(input_file, "rt", encoding="UTF-8") as f:
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
        print(self.get_chin_train_df()[0].head())
        print(self.get_eng_train_df()[1].head())
        print("\n")

        print("dev data:")
        print(self.get_chin_dev_df()[0].head())
        print(self.get_eng_dev_df()[1].head())
        print("\n")

        print("test data:")
        print(self.get_chin_test_df()[0].head())
        print(self.get_eng_test_df()[1].head())
        print("\n")

        print("Train number:{}, Dev number:{}, Test number:{}".format(self.train_num, self.dev_num, self.test_num))

    def set_dir(self, dataset_dir):
        self.dataset_dir = dataset_dir
