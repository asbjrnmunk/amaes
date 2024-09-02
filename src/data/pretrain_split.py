import logging
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle
from yucca.pipeline.configuration.configure_paths import PathConfig
from yucca.pipeline.configuration.split_data import SplitConfig, simple_split, split_is_precomputed, get_file_names


def get_pretrain_split_config(method: str, idx: int, split_ratio: float, path_config: PathConfig):
    splits_path = join(path_config.task_dir, "splits.pkl")

    assert method in [
        "simple_train_val_split",
        "multi_sequence_simple_train_val_split",
    ], "this module only supports a subset of the split methods"

    if isfile(splits_path):
        splits = load_pickle(splits_path)
        assert isinstance(splits, dict)

        if split_is_precomputed(splits, method, idx):
            logging.warning(
                f"Reusing already computed split file which was split using the {method} method and parameter {split_ratio}."
            )
            return SplitConfig(splits, method, idx)
        else:
            logging.warning("Generating new split since splits did not contain a split computed with the same parameters.")
    else:
        splits = {}

    if method not in splits.keys():
        splits[method] = {}

    assert method == "simple_train_val_split"
    names = get_file_names(path_config.train_data_dir)

    splits[method][split_ratio] = simple_split(names, split_ratio)  # type: ignore

    split_cfg = SplitConfig(splits, method, split_ratio)
    save_pickle(splits, splits_path)

    return split_cfg


def get_sequence_names(train_data_dir: str) -> list:
    filenames = subfiles(train_data_dir, join=False, suffix=".npy")
    groups = group_filenames(filenames)

    names = []

    # flatten the dict to list of dicts with metadata useful for debugging
    for dataset in groups.keys():
        for subject in groups[dataset].keys():
            for session in groups[dataset][subject].keys():
                names.append(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "session": session,
                        "filenames": groups[dataset][subject][session],
                    }
                )

    return names
