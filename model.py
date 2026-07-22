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

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # Extract the column of interest
    col = features[:, feature_index]
    # Mask for rows where the feature value is <= threshold (left side)
    mask = col <= threshold
    # Split the data
    left_features = features[mask]
    left_labels = labels[mask]
    right_features = features[~mask]
    right_labels = labels[~mask]
    # Return in the required order
    return (left_features, left_labels, right_features, right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # Compute impurities
    parent_imp = impurity(parent_labels)
    left_imp = impurity(left_labels)
    right_imp = impurity(right_labels)
    
    # Size weights
    n = len(parent_labels)
    w_left = len(left_labels) / n
    w_right = len(right_labels) / n
    
    # Weighted average child impurity
    weighted_child_imp = w_left * left_imp + w_right * right_imp
    
    # Gain (positive when children are purer)
    return parent_imp - weighted_child_imp

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

