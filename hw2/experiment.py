import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from aggregation import (
    estimate_weights,
    fit_dawid_skene,
    majority_vote,
    weighted_majority_vote,
)
from dataset import load_rte

SPLIT_SEED = 42
CALIBRATION_PER_CLASS = 80
BOOTSTRAP_SEED = 43
BOOTSTRAP_SAMPLES = 5_000


def split_tasks(gold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    calibration = []
    for label in (0, 1):
        candidates = rng.permutation(np.flatnonzero(gold == label))
        calibration.extend(candidates[:CALIBRATION_PER_CLASS])
    training = np.sort(calibration)
    testing = np.setdiff1d(np.arange(len(gold)), training)
    return training, testing


def describe_votes(votes: np.ndarray, gold: np.ndarray) -> dict:
    zero = (votes == 0).sum(axis=1)
    one = (votes == 1).sum(axis=1)
    total = zero + one
    workload = (votes != -1).sum(axis=0)
    splits = Counter(zip(np.maximum(zero, one), np.minimum(zero, one)))
    # Ordered pairs give the same agreement rate as unordered pairs.
    agreement = (
        (zero * (zero - 1) + one * (one - 1)) / (total * (total - 1))
    ).mean()
    positive_share = one.sum() / total.sum()
    chance = positive_share**2 + (1 - positive_share) ** 2
    return {
        "tasks": len(gold),
        "workers": votes.shape[1],
        "votes": int(total.sum()),
        "gold_counts": np.bincount(gold, minlength=2).tolist(),
        "min_votes_per_task": int(total.min()),
        "max_votes_per_task": int(total.max()),
        "worker_load_min_median_max": [
            int(workload.min()),
            float(np.median(workload)),
            int(workload.max()),
        ],
        "workers_with_minimum_load": int((workload == workload.min()).sum()),
        "top_five_workers_votes": int(np.sort(workload)[-5:].sum()),
        "positive_votes": int(one.sum()),
        "vote_splits": {
            f"{high}:{low}": count
            for (high, low), count in sorted(splits.items(), reverse=True)
        },
        "pairwise_agreement": float(agreement),
        "fleiss_kappa": float((agreement - chance) / (1 - chance)),
    }


def quality(gold: np.ndarray, predicted: np.ndarray) -> dict:
    confusion = np.bincount(2 * gold + predicted, minlength=4).reshape(2, 2)
    support = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(
        2 * confusion.diagonal(), support, out=np.zeros(2), where=support > 0
    )
    return {
        "correct": int(confusion.trace()),
        "accuracy": float(confusion.trace() / len(gold)),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
    }


def paired_differences(
    gold: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict:
    """Compare accuracy on the same bootstrap sample for each method."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for label in (0, 1):
        group = np.flatnonzero(gold == label)
        samples.append(rng.choice(group, size=(BOOTSTRAP_SAMPLES, len(group))))
    resampled = np.column_stack(samples)
    result = {}
    for baseline, method in combinations(predictions, 2):
        delta = (predictions[method] == gold).astype(int) - (
            predictions[baseline] == gold
        )
        bootstrap = delta[resampled].mean(axis=1)
        result[f"{method} minus {baseline}"] = {
            "gains": int((delta == 1).sum()),
            "losses": int((delta == -1).sum()),
            "difference": float(delta.mean()),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        }
    return result


def run_experiment(votes: np.ndarray, gold: np.ndarray) -> dict:
    training, testing = split_tasks(gold)
    calibration, evaluation = votes[training], votes[testing]
    weights = estimate_weights(calibration, gold[training])
    model = fit_dawid_skene(calibration)
    majority = majority_vote(evaluation)
    predictions = {
        "Majority Vote": majority,
        "Weighted Majority Vote": weighted_majority_vote(evaluation, weights),
        "Dawid–Skene": model.predict(evaluation),
    }

    reference = gold[testing]
    zero = (evaluation == 0).sum(axis=1)
    one = (evaluation == 1).sum(axis=1)
    majority_size = np.maximum(zero, one)
    tied = zero == one
    majority_by_support = {}
    for size in np.unique(majority_size):
        group = majority_size == size
        majority_by_support[str(size)] = {
            "tasks": int(group.sum()),
            "errors": int((majority[group] != reference[group]).sum()),
        }
    return {
        "dataset": describe_votes(votes, gold),
        "protocol": {
            "split_seed": SPLIT_SEED,
            "calibration_per_class": CALIBRATION_PER_CLASS,
            "calibration_tasks": len(training),
            "test_tasks": len(testing),
            "test_gold_counts": np.bincount(reference, minlength=2).tolist(),
            "workers_absent_from_calibration": int(
                (calibration == -1).all(axis=0).sum()
            ),
            "em_iterations": model.iterations,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "individual_vote_accuracy_on_test": float(
            (evaluation == reference[:, None]).sum() / (evaluation != -1).sum()
        ),
        "quality": {
            name: quality(reference, pred)
            for name, pred in predictions.items()
        },
        "paired_accuracy": paired_differences(reference, predictions),
        "test_majority_by_support": majority_by_support,
        "majority_expected_accuracy_with_random_ties": float(
            np.where(tied, 0.5, majority == reference).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the RTE-1 crowd judgments."
    )
    base = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=base / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    votes, gold = load_rte(args.data_dir)
    results = run_experiment(votes, gold)
    serialized = json.dumps(
        results, ensure_ascii=False, indent=2, allow_nan=False
    )
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
