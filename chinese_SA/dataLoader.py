import codecs
import csv
import os

import pandas as pd
import tensorflow as tf

from google_trans_new import google_translator

URL = "https://paddlehub-dataset.bj.bcebos.com/chnsenticorp.tar.gz"

class ChnSentiCorp:
    def __init__(self):
        self.dataset_dir = os.path.join(".", "datasets/chnsenticorp")
        file_path = tf.keras.utils.get_file(
            fname="chnsenticorp.tar.gz",
            origin=URL,
            extract=True,
            cache_dir=".",
        )
        self.translator = google_translator()
        self.train_file, self.chin_train_df, self.eng_train_df, self.train_num = None, None, None, None
        self.dev_file, self.chin_dev_df, self.eng_dev_df, self.dev_num = None, None, None, None
        self.test_file, self.chin_test_df, self.eng_test_df, self.test_num = None, None, None, None

        self.load_data()

    def load_data(self):
        """
        Update the variables with the input
        :return:
        """

        print("load training data")
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.chin_train_df, self.eng_train_df = self.get_df_from_file(self.train_file)
        self.train_num = len(self.chin_train_df)

        print("load dev data")
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.chin_dev_df, self.eng_dev_df = self.get_df_from_file(self.dev_file)
        self.dev_num = len(self.chin_dev_df)

        print("load test data")
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.chin_test_df, self.eng_test_df = self.get_df_from_file(self.test_file)
        self.chin_test_df["labels"] = self.chin_test_df["labels"].apply(lambda x: x[0])
        self.eng_test_df["labels"] = self.eng_test_df["labels"].apply(lambda x: x[0])
        self.test_num = len(self.chin_dev_df)

        print("loading done")

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
            chin_data = []
            eng_data = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                translation = self.translator.translate(line[1].encode('unicode_escape'), lang_tgt='en')
                chin_data.append([line[1], line[0]])
                eng_data.append([translation, line[0]])

                seq_id += 1
            chin_df = pd.DataFrame(chin_data)
            eng_df = pd.DataFrame(eng_data)
            chin_df.columns = ["text", "labels"]
            chin_df["labels"] = chin_df["labels"].apply(lambda x: list(map(int, x)))

            eng_df.columns = ["text", "labels"]
            eng_df["labels"] = eng_df["labels"].apply(lambda x: list(map(int, x)))
            return chin_df, eng_df

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
