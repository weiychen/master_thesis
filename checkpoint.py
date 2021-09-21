import os
import pandas as pd
from abc import ABC, abstractmethod

import torch
from sdv.tabular.ctgan import CTGAN

import logger
import config

class ISaveLoad(ABC):
    """ The ISaveLoad Interface defines the 'save' and 'load' methods
    needed by a Checkpoint. The checkpoint uses these to load and save
    models with the model specific algorithms. """
    @abstractmethod
    def save(self, obj: object, path: str):
        """ Save the obj to path """

    @abstractmethod
    def load(self, path: str):
        """ Save the obj to path """

class Checkpoint:
    """ A Checkpoint stores models as a file, where the file name contains hyperparameters
    for later identification. 
    To add hyperparameters to add to the filename, the 'add_info' method can be used.
    The 'exists' method can be used to check if a file with the hyperparameters of the
    checkpoint exists, which can then be loaded using the 'load' method.
    The 'save' method saves the model with the hyperparameters of the checkpoint.
    """

    def __init__(self, folder_path: str, saveload: ISaveLoad, model_name: str = "mdl", file_type: str = ".cp"):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True) # create folder if not exists
        self.model_name: str = model_name
        self.file_type: str = file_type
        self.saveload: ISaveLoad = saveload
        self.save_file: str = model_name
        self.infos: list = list()
        self.set_name(None)

    def add_info(self, name: str, value):
        """ Add info to be included in the Checkpoint's file name.
        The format of this in the file name is 'name-value' where different
        info's are separated by an underscore.
        """
        self.infos.append({"name":name, "value": value})
        self._build_savefile_name()

    def set_name(self, name: str):
        """ Give the checkpoint a name, which is printed when loading. """
        self.checkpoint_name = name

    def _build_savefile_name(self):
        self.save_file = os.path.join(self.folder_path, self.model_name)
        for info in self.infos:
            value = info["value"]
            name = info["name"]
            self.save_file += f"_{name}-{value}"
        self.save_file += self.file_type

    def exists(self) -> bool:
        """ Return True if a checkpoint file exists, otherwise return False. """
        return os.path.exists(self.save_file)

    def save(self, obj, override=True):
        """ Save obj to the model file.
        Nothing is done if the override flag is not set.
        """
        if not self.exists() or override:
            self.saveload.save(obj, self.save_file)

    def load(self):
        if self.exists():
            return self.saveload.load(self.save_file)

    def load_if_exists_else_generate(self, force_generate, generate_func, *args, **kwargs):
        """ Load checkpoint from file or generate a new one if file doesn't exist. """
        name = " '" + self.checkpoint_name + "'" if self.checkpoint_name else ""
        if self.exists() and not force_generate:
            logger.log(f"Loading checkpoint{name} from '{self.save_file}'", summary=True)
            return self.load()
        else:
            logger.log(f"Checkpoint{name} file does not exist: {self.save_file}", summary=True)
            logger.log(f"Regenerating at checkpoint{name}...", summary=True)
            generated = generate_func(*args, **kwargs)
            self.save(generated)
            return generated


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
    def __init__(self, dataset_name, epochs, enabled_dp, epsilon):
        path = os.path.join(config.CHECKPOINTS_ROOT, "ctgan")
        super().__init__(path, CTGANSaveLoad(), "ctgan", ".mdl")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("dp", enabled_dp)
        self.add_info("eps", epsilon)

        self.set_name("CTGAN Train")

class ResultsCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, enabled_dp):
        path = os.path.join(config.CHECKPOINTS_ROOT, "results")
        super().__init__(path, DataframeSaveLoad(), "result", ".csv")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("dp", enabled_dp)

        self.set_name("Results")

class LSTMCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, epsilon: str):
        path = os.path.join(config.CHECKPOINTS_ROOT, "lstm", "nn_models")
        super().__init__(path, LSTMSaveLoad(), "lstm")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("eps", epsilon)

        self.set_name("nn.Model")

class GeneratedWordsCheckpoint(Checkpoint):
    def __init__(self, dataset_name, epochs, epsilon: str):
        path = os.path.join(config.CHECKPOINTS_ROOT, "lstm", "generated_words_dfs")
        super().__init__(path, DataframeSaveLoad(), "words", ".csv")
        self.add_info("dataset", dataset_name)
        self.add_info("epochs", epochs)
        self.add_info("eps", epsilon)

        self.set_name("words")