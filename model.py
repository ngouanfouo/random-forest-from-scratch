"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # Handle edge cases: empty array or single element
    if len(labels) <= 1:
        return 0.0
    
    # Count occurrences of each class
    unique_values, counts = np.unique(labels, return_counts=True)
    
    # If only one unique class, it's pure
    if len(unique_values) == 1:
        return 0.0
    
    # Calculate proportions
    n = len(labels)
    proportions = counts / n
    
    # Gini impurity: 1 - sum(p_k^2)
    gini = 1.0 - np.sum(proportions ** 2)
    
    return float(gini)

# Step 2 - split_dataset (not yet solved)
# TODO: implement

# Step 3 - split_score (not yet solved)
# TODO: implement

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

