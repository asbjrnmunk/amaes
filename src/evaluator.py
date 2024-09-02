import os
from functools import partial
from git import Optional
import numpy as np
import nibabel as nib
import json
import sys
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, load_json, isfile
from sklearn.metrics import confusion_matrix
from yucca.functional.evaluation.metrics import (
    dice,
    jaccard,
    sensitivity,
    precision,
    TP,
    FP,
    FN,
    total_pos_gt,
    total_pos_pred,
    volume_similarity,
)
from yucca.functional.evaluation.surface_metrics import get_surface_metrics_for_label
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class Evaluator:
    def __init__(
        self,
        labels: list | int,
        folder_with_predictions,
        folder_with_ground_truth,
        raw_data_path,
        as_binary=False,
        do_surface_eval=False,
        overwrite: bool = False,
        num_workers=4,
        ignore_labels=["0"],
        include_cases: Optional[list] = None,
    ):
        self.name = "results"

        self.overwrite = overwrite
        self.surface_metrics = []

        self.metrics = {
            "Dice": dice,
            "Jaccard": jaccard,
            "Sensitivity": sensitivity,
            "Precision": precision,
            "Volume Similarity": volume_similarity,
            "True Positives": TP,
            "False Positives": FP,
            "False Negatives": FN,
            "Total Positives Ground Truth": total_pos_gt,
            "Total Positives Prediction": total_pos_pred,
        }

        if do_surface_eval:
            self.name += "_SURFACE"
            self.surface_metrics = [
                "Average Surface Distance",
            ]

        if isinstance(labels, int):
            self.labels = [str(i) for i in range(labels)]
        else:
            self.labels = labels
        self.as_binary = as_binary
        if self.as_binary:
            self.labels = ["0", "1"]
            self.name += "_BINARY"

        self.labels = np.sort(np.array(self.labels, dtype=np.uint8))
        self.ignore_labels = ignore_labels
        self.folder_with_predictions = folder_with_predictions
        self.folder_with_ground_truth = folder_with_ground_truth
        self.raw_data_path = raw_data_path

        self.outpath = join(self.folder_with_predictions, f"{self.name}.json")

        self.pred_subjects = subfiles(self.folder_with_predictions, suffix=".nii.gz", join=False)
        gt_subjects = subfiles(self.folder_with_ground_truth, suffix=".nii.gz", join=False)
        if include_cases is not None:
            include_cases_with_suffix = [f + ".nii.gz" for f in include_cases]
            self.gt_subjects = [f for f in gt_subjects if f in include_cases_with_suffix]
        else:
            self.gt_subjects = gt_subjects

        self.num_workers = num_workers

        print(
            f"\n"
            f"STARTING EVALUATION \n"
            f"Folder with predictions: {self.folder_with_predictions}\n"
            f"Folder with ground truth: {self.folder_with_ground_truth}\n"
            f"Evaluating performance on labels: {self.labels}"
        )

    def sanity_checks(self):
        print("pred subjects", self.pred_subjects)
        print("gt subjects", self.gt_subjects)

        assert self.pred_subjects <= self.gt_subjects, "Ground Truth is missing for predicted scans"

        assert self.gt_subjects <= self.pred_subjects, "Prediction is missing for Ground Truth of scans"

        # Check if the Ground Truth directory is a subdirectory of a 'TaskXXX_MyTask' folder.
        # If so, there should be a dataset.json where we can double check that the supplied classes
        # match with the expected classes for the dataset.
        gt_is_task = [i for i in self.folder_with_ground_truth.split(os.sep) if "Task" in i]
        if gt_is_task:
            gt_task = gt_is_task[0]
            dataset_json = join(self.raw_data_path, gt_task, "dataset.json")
            if isfile(dataset_json):
                dataset_json = load_json(dataset_json)
                print(f"Labels found in dataset.json: {list(dataset_json['labels'].keys())}")

    def run(self):
        if isfile(self.outpath) and not self.overwrite:
            print(f"Evaluation file already present in {self.outpath}. Skipping.")
        else:
            self.sanity_checks()
            results_dict = self.evaluate_folder()
            self.save_as_json(results_dict)

    def evaluate_folder(self):
        sys.stdout.flush()

        metric_names = list(self.metrics.keys()) + self.surface_metrics

        map_func = partial(
            process_case,
            pred_dir=self.folder_with_predictions,
            gt_dir=self.folder_with_ground_truth,
            labels=self.labels,
            metrics=self.metrics,
            calc_surface_metrics=len(self.surface_metrics) > 0,
            as_binary=self.as_binary,
            ignore_labels=self.ignore_labels,
        )

        results = process_map(
            map_func,
            self.pred_subjects,
            max_workers=self.num_workers,
            chunksize=1,
            desc="Map",
        )

        all_labels = [str(label) for label in self.labels] + ["all"]

        aggregated_results = reduce(results, labels=all_labels, metric_names=metric_names)

        mean_results = {}

        for label in all_labels:
            mean_results[str(label)] = {
                metric: round(np.nanmean(metric_vals), 4) if not np.all(np.isnan(metric_vals)) else 0
                for metric, metric_vals in aggregated_results[str(label)].items()
            }

        return dict(results) | {"mean": mean_results}

    def save_as_json(self, dict):
        print("Saving results.json at path: ", self.outpath)
        with open(self.outpath, "w") as f:
            json.dump(dict, f, default=float, indent=4)


def process_case(case, pred_dir, gt_dir, labels, metrics, calc_surface_metrics, as_binary, ignore_labels):
    pred_path = join(pred_dir, case)
    gt_path = join(gt_dir, case)

    results = {}

    pred = nib.load(pred_path)
    gt = nib.load(gt_path)

    if as_binary:
        cmat = confusion_matrix(
            np.around(gt.get_fdata().flatten()).astype(bool).astype(int),
            np.around(pred.get_fdata().flatten()).astype(bool).astype(int),
            labels=labels,
        )
    else:
        cmat = confusion_matrix(
            np.around(gt.get_fdata().flatten()).astype(int),
            np.around(pred.get_fdata().flatten()).astype(int),
            labels=labels,
        )

    tp_agg = 0
    fp_agg = 0
    fn_agg = 0
    tn_agg = 0

    for label in labels:
        label_results = {}

        tp = cmat[label, label]
        fp = sum(cmat[:, label]) - tp
        fn = sum(cmat[label, :]) - tp
        tn = np.sum(cmat) - tp - fp - fn  # often a redundant and meaningless metric

        label_str = str(label)
        if label_str not in ignore_labels:
            tp_agg += tp
            fp_agg += fp
            fn_agg += fn
            tn_agg += tn

        for metric, metric_function in metrics.items():
            label_results[metric] = round(metric_function(tp, fp, tn, fn), 4)

        if calc_surface_metrics:
            surface_metrics = get_surface_metrics_for_label(gt, pred, label, as_binary=as_binary)
            for surface_metric, val in surface_metrics.items():
                label_results[surface_metric] = round(val, 4)

        results[str(label)] = label_results

    results["all"] = {
        metric: round(metric_function(tp_agg, fp_agg, tn_agg, fn_agg), 4) for metric, metric_function in metrics.items()
    }

    # results["Prediction:"] = predpath
    # results["Ground Truth:"] = gtpath

    # `results` contains
    # {
    #   "0": { "dice": 0.8, "f1": 0.9 }
    #  ...
    #  }
    return (case, results)


def reduce(results_per_case, labels, metric_names):
    results_per_label = {}

    for label in labels:
        results_per_label[str(label)] = {}
        for metric in metric_names:
            results_per_label[str(label)][metric] = []

    for _, results in tqdm(results_per_case, "reduce"):
        for label, label_result in results.items():
            for metric, metric_val in label_result.items():
                results_per_label[str(label)][metric].append(metric_val)

    # `results_per_label` contains
    # {
    #   "0": {
    #     "dice": [0.8, 0.9, 0.2, ...]
    #     ...
    #   }
    # }
    return results_per_label
