import os
import tarfile

from pandas.core.frame import DataFrame
from simpletransformers.classification import ClassificationModel
from chinese_SA.dataLoader import DataLoader

class Model():
    def __init__(self, model_type: str, model_name: str):
        self.model_type = model_type
        self.model_name = model_name
        self.model_path = "outputs"
        self.model = None
        self.train_df = None

    def train_model(self, train_df, num_labels):
        # define hyperparameter
        train_args ={"reprocess_input_data": True,
                     "overwrite_output_dir": True,
                     "fp16": False,
                     "num_train_epochs": 4}

        # Create a ClassificationModel
        model = ClassificationModel(
            self.model_type, self.model_name,
            num_labels= num_labels,
            args=train_args
        )
        model.train_model(train_df)

        self.model = model
        self._save_model()

    def is_trained(self) -> bool:
        return self.model is not None

    def is_saved(self) -> bool:
        return os.path.exists(f"{self.model_name}.tar.gz", "r:gz")

    def _save_model(self):
        files = [files for root, dirs, files in os.walk(self.model_path)][0]
        with tarfile.open(self.model_name + '.tar.gz', 'w:gz') as f:
            for file in files:
                f.add(f'{self.model_path}/{file}')

    def eval_model(self, test_df):
        result, model_outputs, wrong_predictions = self.model.eval_model(test_df)
        accuracy = (result['tp'] + result['tn'])/len(test_df)
        return accuracy