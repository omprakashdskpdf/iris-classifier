"""Rebuild iris_model.json from scikit-learn's Iris dataset."""
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.20, random_state=42, stratify=iris.target
)
clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
t = clf.tree_
nodes = []
for i in range(t.node_count):
    left, right = int(t.children_left[i]), int(t.children_right[i])
    leaf = left == right
    counts = [int(v) for v in t.value[i][0]]
    total = sum(counts)
    nodes.append({
        "id": i, "is_leaf": leaf,
        "feature_index": None if leaf else int(t.feature[i]),
        "feature_name": None if leaf else iris.feature_names[int(t.feature[i])],
        "threshold": None if leaf else float(t.threshold[i]),
        "left_child": None if leaf else left,
        "right_child": None if leaf else right,
        "class_counts": counts,
        "probabilities": [round(v / total, 8) if total else 0 for v in counts],
        "predicted_class_index": int(np.argmax(counts))
    })
out = {
    "model_name": "Iris Decision Tree Classifier",
    "model_type": "decision_tree_classifier", "version": "1.0",
    "input_features": iris.feature_names,
    "class_names": [str(x) for x in iris.target_names],
    "root_node": 0, "nodes": nodes
}
with open("iris_model.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Saved iris_model.json")
