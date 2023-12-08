import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def compute_multiclass_auc(y_true, y_scores):
    """
    Compute AUC for each class separately in a multi-class problem.

    Parameters:
    - y_true: true labels (array-like of shape (n_samples,))
    - y_scores: predicted scores (array-like of shape (n_samples, n_classes))

    Returns:
    - auc_scores: AUC scores for each class (array-like of shape (n_classes,))
    """
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)

    auc_scores = []
    for i in range(y_scores.shape[1]):
        auc_i = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        auc_scores.append(auc_i)

    return np.array(auc_scores)
