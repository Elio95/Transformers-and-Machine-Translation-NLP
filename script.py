import os
import tarfile

from dataModels.mixImdbChin import MixedDatasets
from models.model import Model

models = {
    ('bert', "bert-base-chinese"): None,
}


def unpack_model(model_name):
    tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
    tar.extractall()
    tar.close()


mix = MixedDatasets()
mix.print_info()
