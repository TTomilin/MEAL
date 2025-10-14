import json
import os
import pickle
import tempfile
import uuid
from pathlib import Path


def save_pickle(data, filename):
    with open(fix_filetype(filename, ".pickle"), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(fix_filetype(filename, ".pickle"), "rb") as f:
        return pickle.load(f)


def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())


def save_dict_to_file(dic, filename):
    dic = dict(dic)
    with open(fix_filetype(filename, ".txt"), "w") as f:
        f.write(str(dic))


def load_dict_from_txt(filename):
    return load_dict_from_file(fix_filetype(filename, ".txt"))


def save_as_json(data, filename):
    with open(fix_filetype(filename, ".json"), "w") as outfile:
        json.dump(data, outfile)
    return filename


def load_from_json(filename):
    with open(fix_filetype(filename, ".json"), "r") as json_file:
        return json.load(json_file)


def iterate_over_json_files_in_dir(dir_path):
    pathlist = Path(dir_path).glob("*.json")
    return [str(path) for path in pathlist]


def fix_filetype(path, filetype):
    if path[-len(filetype):] == filetype:
        return path
    else:
        return path + filetype


def generate_temporary_file_path(
        file_name=None, prefix="", suffix="", extension=""
):
    if file_name is None:
        file_name = str(uuid.uuid1())
    if extension and not extension.startswith("."):
        extension = "." + extension
    file_name = prefix + file_name + suffix + extension
    return os.path.join(tempfile.gettempdir(), file_name)
