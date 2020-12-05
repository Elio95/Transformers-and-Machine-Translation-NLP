import tarfile

from chinese_SA.dataLoader import DataLoader
from chinese_SA.model import Model

models = {
    ('bert', "bert-base-chinese"): None,
}


def unpack_model(model_name):
    tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()


data = DataLoader()
for model_tuple in models:
    model = Model(*model_tuple, data)
    model.train_model()
    models[model_tuple] = model

