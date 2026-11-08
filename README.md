# Iris Species Classifier — Streamlit

A public Streamlit web app demonstrating inference from a trained **JSON decision-tree model**.

## Files

- `app.py` — Streamlit front end and JSON model inference.
- `iris_model.json` — trained Iris decision-tree structure and leaf probabilities.
- `metadata.json` — feature labels, input ranges, classes and model information.
- `requirements.txt` — Python dependencies for Streamlit Community Cloud.
- `train_model.py` — optional script to regenerate the JSON model.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly with GitHub + Streamlit Community Cloud

1. Create a new **public** GitHub repository, e.g. `iris-streamlit-classifier`.
2. Upload all files in this folder to the repository root (not inside an extra nested folder).
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will install packages from `requirements.txt` and provide a public app URL.

## Replace the demo model with your own JSON model

The current app expects a JSON decision tree with this general structure:

```json
{
  "model_type": "decision_tree_classifier",
  "class_names": ["class_A", "class_B"],
  "root_node": 0,
  "nodes": [
    {
      "id": 0,
      "is_leaf": false,
      "feature_index": 0,
      "threshold": 2.5,
      "left_child": 1,
      "right_child": 2,
      "probabilities": [0.9, 0.1],
      "predicted_class_index": 0
    }
  ]
}
```

A standard TensorFlow/Keras `model.json` is **not** the same format as this custom JSON tree. If your Colab model is TensorFlow/Keras, it normally needs its accompanying weight files and a different inference loader (or conversion to TensorFlow Lite / another deployable format). A scikit-learn `.pkl`/`.joblib` file also needs a different loader. Do not simply rename those files to `.json`.

For a different model, keep the UI pattern but adapt `predict_one()` and the model-loading function to the actual model format and exact input feature order. Never load untrusted pickle/joblib files from unknown sources.

## Demo model details

- Dataset: scikit-learn Iris (150 rows, 4 numeric measurements, 3 classes).
- Algorithm: `DecisionTreeClassifier(max_depth=4)`.
- Stratified held-out test set: 20%.
- Held-out accuracy: **93.3%** (random state 42).

This is an educational demonstration, not a production-grade scientific validation.
