import os
import pandas as pd
from abc import ABC, abstractmethod

import torch
from sdv.tabular.ctgan import CTGAN

class ISaveLoad(ABC):
    @abstractmethod
    def save(self, obj: object, path: str):
        """ Save the obj to path """

    @abstractmethod
    def load(self, path: str):
        """ Save the obj to path """

class Checkpoint:
    def __init__(self, folder_path: str, saveload: ISaveLoad, model_name: str = "mdl", file_type: str = ".cp"):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True) # create folder if not exists
        self.model_name: str = model_name
        self.file_type: str = file_type
        self.saveload: ISaveLoad = saveload
        self.save_file: str = model_name
        self.infos: list = list()

    def add_info(self, name: str, value):
        self.infos.append({"name":name, "value": value})
        self._build_savefile_name()

    def _build_savefile_name(self):
        self.save_file = os.path.join(self.folder_path, self.model_name)
        for info in self.infos:
            value = info["value"]
            name = info["name"]
            self.save_file += f"_{name}-{value}"
        self.save_file += self.file_type

    def exists(self):
        return os.path.exists(self.save_file)

    def save(self, obj, override=True):
        if not self.exists() or override:
            self.saveload.save(obj, self.save_file)

    def load(self):
        if self.exists():
            return self.saveload.load(self.save_file)


class CTGANSaveLoad(ISaveLoad):
    def save(self, obj: CTGAN, path: str):
        obj.save(path)
    def load(self, path: str):
        return CTGAN.load(path)

class DataframeSaveLoad(ISaveLoad):
    def save(self, obj: pd.DataFrame, path: str):
        obj.to_csv(path)
    def load(self, path: str):
        return pd.read_csv(path, index_col=0)

class LSTMSaveLoad(ISaveLoad):
    def save(self, obj: torch.nn.Module, path: str):
        torch.save(obj, path)
    def load(self, path: str):
        return torch.load(path)


class CTGANCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, enabled_dp):
        super().__init__("fitted_models", CTGANSaveLoad(), "ctgan", ".mdl")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("dp", enabled_dp)

class DataframeCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, enabled_dp):
        super().__init__("results", DataframeSaveLoad(), "sampled", ".csv")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("dp", enabled_dp)

class LSTMCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, epsilon: str):
        super().__init__("nn_models", LSTMSaveLoad(), "lstm")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("eps", epsilon)