import os

class Checkpoint:
    def __init__(self, folder_path: str, model_name: str = "mdl"):
        self.folder_path = folder_path
        os.makedirs(folder_path, exist_ok=True) # create folder if not exists
        self.model_name: str = model_name
        self.infos: list = list()
        self.save_file: str = model_name

    def add_info(self, name: str, value):
        self.infos.append({"name":name, "value": value})
        self._build_savefile_name()

    def _build_savefile_name(self):
        self.save_file = os.path.join(self.folder_path, self.model_name)
        for info in self.infos:
            value = info["value"]
            name = info["name"]
            self.save_file += f"_{name}-{value}"

    def exists(self):
        return os.path.exists(self.save_file)

    def save(self, obj, save_function, override=False):
        if not self.exists() or override:
            save_function(obj, self.save_file)

    def load(self, load_function):
        if self.exists():
            return load_function(self.save_file)
