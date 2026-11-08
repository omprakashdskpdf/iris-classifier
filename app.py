import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "iris_model.json"
META_PATH = ROOT / "metadata.json"

st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="wide")

@st.cache_data
def load_assets():
    with MODEL_PATH.open("r", encoding="utf-8") as f:
        model = json.load(f)
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def predict_one(model_data, values):
    """Traverse the JSON decision tree and return class probabilities."""
    node_map = {node["id"]: node for node in model_data["nodes"]}
    node = node_map[model_data.get("root_node", 0)]
    while not node["is_leaf"]:
        feature_value = float(values[node["feature_index"]])
        if feature_value <= float(node["threshold"]):
            node = node_map[node["left_child"]]
        else:
            node = node_map[node["right_child"]]
    probs = node["probabilities"]
    class_index = int(node["predicted_class_index"])
    return class_index, probs

model_data, meta = load_assets()
classes = model_data["class_names"]
labels = meta["feature_labels"]
ranges = meta["feature_ranges"]

st.title("🌸 Iris Species Classifier")
st.caption("An interactive machine-learning demonstration powered by a trained JSON decision-tree model.")

with st.sidebar:
    st.header("About this model")
    st.metric("Held-out test accuracy", f'{meta["test_accuracy"] * 100:.1f}%')
    st.write(f'**Algorithm:** {meta["algorithm"]}')
    st.write(f'**Test samples:** {meta["test_size"]}')
    st.info("This is a demonstration model. Predictions are based on the supplied measurements and should not be used for any clinical or high-stakes decision.")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("Enter flower measurements")
    with st.form("prediction_form"):
        values = []
        cols = st.columns(2)
        for i, feature in enumerate(meta["features"]):
            r = ranges[i]
            with cols[i % 2]:
                value = st.number_input(
                    labels[i],
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=float(r["default"]),
                    step=float(r["step"]),
                    key=f"feature_{i}"
                )
                values.append(value)
        submitted = st.form_submit_button("Predict Iris species", type="primary", use_container_width=True)

    st.markdown("#### Quick examples")
    sample_cols = st.columns(3)
    samples = [
        ("Setosa-like", [5.1, 3.5, 1.4, 0.2]),
        ("Versicolor-like", [6.0, 2.7, 4.5, 1.5]),
        ("Virginica-like", [6.5, 3.0, 5.5, 2.0])
    ]
    for col, (name, sample) in zip(sample_cols, samples):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state["sample_values"] = sample
                st.rerun()

    if "sample_values" in st.session_state and not submitted:
        st.caption("Example selected. Copy its measurements into the inputs above to run a prediction.")

with right:
    st.subheader("Prediction")
    if submitted:
        class_index, probabilities = predict_one(model_data, values)
        predicted_name = classes[class_index]
        st.success(f"Predicted species: **{predicted_name.title()}**")
        st.write("Class probabilities at the reached decision-tree leaf:")
        probability_df = pd.DataFrame({
            "Species": [name.title() for name in classes],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.bar_chart(probability_df.set_index("Species"))
        st.dataframe(
            probability_df.assign(Probability=probability_df["Probability"].map(lambda x: f"{x:.1%}")),
            hide_index=True,
            use_container_width=True
        )
        st.write("**Input values**")
        st.dataframe(pd.DataFrame({"Measurement": labels, "Value (cm)": values}), hide_index=True, use_container_width=True)
    else:
        st.info("Adjust the four measurements and select **Predict Iris species** to see the model output.")
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/640px-Iris_versicolor_3.jpg",
            caption="Iris flowers (illustrative image)",
            use_container_width=True
        )

st.divider()
st.subheader("How to use this app")
st.markdown("""
1. Enter sepal and petal measurements in centimetres.
2. Select **Predict Iris species**.
3. Review the predicted class and the probability distribution at the reached tree leaf.

**Deployment:** Push this folder to a GitHub repository, then create a Streamlit Community Cloud app pointing to `app.py`.
""")
