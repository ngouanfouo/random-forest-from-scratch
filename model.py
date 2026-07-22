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

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_score = 0.0
    best_feature = None
    best_threshold = None
    
    n_samples = len(labels)
    
    for feature_idx in feature_indices:
        # Get the column values
        col = features[:, feature_idx]
        
        # Get unique values in sorted order
        unique_values = np.unique(col)
        
        # Candidate thresholds are midpoints between consecutive unique values
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2.0
            
            # Split the data
            mask = col <= threshold
            left_labels = labels[mask]
            right_labels = labels[~mask]
            
            # Skip if either side is empty
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            # Score the split
            score = split_score(labels, left_labels, right_labels)
            
            # Update best if this split is better
            if score > best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold
    
    return {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'score': best_score
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # Check if node is pure (all labels are the same)
    # If impurity is 0, it's pure
    if impurity(labels) == 0.0:
        return True
    
    # Check if we've reached the maximum depth
    # depth is 0-indexed, so if depth >= max_depth, we can't go deeper
    if depth >= max_depth:
        return True
    
    # Check if we have too few samples to split
    if len(labels) < min_samples_split:
        return True
    
    # Otherwise, we can still split
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # Choose the majority class label
    # np.bincount counts occurrences of each class, argmax finds the most frequent
    return int(np.bincount(labels).argmax())

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # Check if we should stop at this node
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    # Determine which features to consider
    if feature_subset is None:
        candidate_features = list(range(features.shape[1]))
    else:
        candidate_features = list(feature_subset)
    
    # If no features to consider, return a leaf
    if len(candidate_features) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    # Find the best split
    split_info = best_split(features, labels, candidate_features)
    
    # If no valid split found, return a leaf
    if split_info['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    feature_idx = split_info['feature_index']
    threshold = split_info['threshold']
    
    # Partition the data
    left_features, left_labels, right_features, right_labels = split_dataset(
        features, labels, feature_idx, threshold
    )
    
    # If either side is empty, return a leaf
    if len(left_labels) == 0 or len(right_labels) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    # Recursively build left and right subtrees
    left_subtree = build_tree(
        left_features, left_labels, 
        max_depth, min_samples_split, 
        feature_subset, depth + 1
    )
    right_subtree = build_tree(
        right_features, right_labels,
        max_depth, min_samples_split,
        feature_subset, depth + 1
    )
    
    # Return an internal node
    return {
        'leaf': False,
        'feature_index': feature_idx,
        'threshold': threshold,
        'left': left_subtree,
        'right': right_subtree
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # Start at the root
    current_node = tree
    
    # Traverse until we hit a leaf
    while not current_node['leaf']:
        # Get the feature index and threshold for this internal node
        feature_idx = current_node['feature_index']
        threshold = current_node['threshold']
        
        # Check which branch to follow
        if example[feature_idx] <= threshold:
            current_node = current_node['left']
        else:
            current_node = current_node['right']
    
    # Return the leaf's prediction
    return int(current_node['prediction'])

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # Handle empty feature matrix
    if len(features) == 0:
        return np.array([], dtype=int)
    
    # Get predictions for each row
    predictions = []
    for row in features:
        predictions.append(predict_example_tree(tree, row))
    
    # Return as numpy array of integers
    return np.array(predictions, dtype=int)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # Get the number of rows
    n = features.shape[0]
    
    # Draw n indices uniformly with replacement
    indices = rng.integers(0, n, size=n)
    
    # Gather the sampled rows for both features and labels
    sampled_features = features[indices]
    sampled_labels = labels[indices]
    
    return sampled_features, sampled_labels

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # Return num_to_pick distinct random feature indices
    # If num_to_pick equals num_features, return all indices
    if num_to_pick >= num_features:
        return np.arange(num_features, dtype=int)
    
    # Use rng.choice with replace=False to get distinct indices
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    # Create a single RNG for reproducibility
    rng = np.random.default_rng(random_state)
    
    # Get number of features
    d = features.shape[1]
    
    # Determine number of features per split
    if num_features_per_split is None:
        num_features_per_split = max(1, int(round(np.sqrt(d))))
    
    # Initialize the forest
    forest = []
    
    # Grow each tree
    for _ in range(num_trees):
        # Draw bootstrap sample of rows
        boot_features, boot_labels = bootstrap_sample(features, labels, rng)
        
        # Draw random subset of features
        feature_indices = feature_subset(d, num_features_per_split, rng)
        
        # Grow a tree using the bootstrap sample and feature subset
        tree = build_tree(
            boot_features, 
            boot_labels,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            feature_subset=feature_indices
        )
        
        # Store the tree and its feature indices
        forest.append({
            'tree': tree,
            'feature_indices': feature_indices
        })
    
    return forest

# Step 13 - combine_predictions
def combine_predictions(tree_predictions):
    # Get number of trees and examples
    num_trees, n_examples = tree_predictions.shape
    
    # Allocate output array
    final_predictions = np.empty(n_examples, dtype=int)
    
    # For each example, find the most common prediction across trees
    for i in range(n_examples):
        # Get all tree predictions for this example
        votes = tree_predictions[:, i]
        
        # Count occurrences of each class
        # Using bincount works if labels are non-negative integers
        counts = np.bincount(votes)
        
        # Find the class with the most votes
        # If there's a tie, argmax returns the first (smallest) class with max count
        final_predictions[i] = np.argmax(counts)
    
    return final_predictions

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

