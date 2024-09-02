from typing import Callable, Dict, List
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_pickle, isfile, load_pickle


def ensure_splits_contains_split(task_dir: str, train_data_dir: str, split_creator: Callable[[List[str], Dict], Dict]):
    splits_path = join(task_dir, "splits.pkl")
    files = subfiles(train_data_dir, join=False, suffix=".npy")
    assert len(files) > 0

    if isfile(splits_path):
        splits = load_pickle(splits_path)
        splits = split_creator(files, splits)
    else:
        splits = split_creator(files, {})

    save_pickle(splits, splits_path)


def create_combination_split_file(files, splits, prefix1, prefix2):
    key = f"combined_{prefix1}_{prefix2}"
    subkey1 = f"train_{prefix1}_val_{prefix2}"
    subkey2 = f"train_{prefix2}_val_{prefix1}"

    if key in splits.keys() and subkey1 in splits[key].keys() and subkey2 in splits[key].keys():
        return splits
    else:
        if key not in splits.keys():
            splits[key] = {}

        split1 = [file for file in files if file.startswith(prefix1)]
        split2 = [file for file in files if file.startswith(prefix2)]
        assert len(files) == len(split1) + len(split2)

        splits[key][subkey1] = [{"train": split1, "val": split2}]
        splits[key][subkey2] = [{"train": split2, "val": split1}]

        return splits
