from dataclasses import dataclass

import numpy as np


def majority_vote(votes: np.ndarray) -> np.ndarray:
    return ((votes == 1).sum(axis=1) > (votes == 0).sum(axis=1)).astype(int)


def estimate_weights(votes: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Estimate nonnegative log-odds weights on calibration tasks."""
    observed = votes != -1
    totals = observed.sum(axis=0)
    correct = (votes == gold[:, None]).sum(axis=0)
    pooled_accuracy = (correct.sum() + 1) / (totals.sum() + 2)
    accuracy = (correct + 5 * pooled_accuracy) / (totals + 5)
    return np.maximum(0, np.log(accuracy / (1 - accuracy)))


def weighted_majority_vote(
    votes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    score_one = (votes == 1) @ weights
    score_zero = (votes == 0) @ weights
    return (score_one > score_zero + 1e-12).astype(int)


@dataclass
class DawidSkene:
    prior: np.ndarray
    confusion: np.ndarray  # worker, true class, reported class
    iterations: int

    def predict_proba(self, votes: np.ndarray) -> np.ndarray:
        task, worker = np.nonzero(votes != -1)
        reported = votes[task, worker]
        scores = np.tile(np.log(self.prior), (len(votes), 1))
        log_likelihood = np.log(self.confusion[worker, :, reported])
        np.add.at(scores, task, log_likelihood)
        return np.exp(
            scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
        )

    def predict(self, votes: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(votes)
        return (probabilities[:, 1] > probabilities[:, 0] + 1e-12).astype(int)


def fit_dawid_skene(votes: np.ndarray) -> DawidSkene:
    """Fit smoothed binary Dawid–Skene using calibration votes alone."""
    task, worker = np.nonzero(votes != -1)
    reported = votes[task, worker]
    unseen = (votes == -1).all(axis=0)
    counts = np.column_stack(
        ((votes == 0).sum(axis=1), (votes == 1).sum(axis=1))
    )
    posterior = (counts + 1) / (counts.sum(axis=1, keepdims=True) + 2)

    for iteration in range(1, 501):
        # The expected counts are fractional while the true class is unknown.
        joint = np.zeros((votes.shape[1], 2, 2))
        np.add.at(joint, (worker, slice(None), reported), posterior[task])
        joint[unseen] = joint.sum(axis=0)
        confusion = (joint + 1) / (joint.sum(axis=2, keepdims=True) + 2)
        prior = (posterior.sum(axis=0) + 1) / (len(votes) + 2)
        model = DawidSkene(prior, confusion, iteration)
        updated = model.predict_proba(votes)
        if np.max(np.abs(updated - posterior)) < 1e-7:
            return model
        posterior = updated

    raise RuntimeError("Dawid–Skene did not converge in 500 iterations")
