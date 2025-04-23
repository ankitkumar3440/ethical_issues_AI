# # Cell 2: app.py logic (Streamlit deployment)

# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import random

# # Reuse generate, preprocess, train, predict from Cell 1
# def generate_sample_data(n=100):
#     data = {
#         "Supplier_ID": [f"S{i+1}" for i in range(n)],
#         "Carbon_Emission": [round(random.uniform(100, 800), 2) for _ in range(n)],
#         "Water_Usage": [round(random.uniform(500, 2000), 2) for _ in range(n)],
#         "Labor_Practices_Score": [random.randint(1, 10) for _ in range(n)],
#         "Certification": [random.choice(["ISO14001", "None", "FairTrade", "LEED"]) for _ in range(n)],
#         "Sustainable": [random.choice([0, 1]) for _ in range(n)]
#     }
#     return pd.DataFrame(data)

# def preprocess_data(df):
#     le = LabelEncoder()
#     df['Certification'] = le.fit_transform(df['Certification'])
#     X = df[['Carbon_Emission', 'Water_Usage', 'Labor_Practices_Score', 'Certification']]
#     y = df['Sustainable']
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# def train_model(X_train, y_train):
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     return model

# def predict(model, input_df):
#     return model.predict(input_df), model.predict_proba(input_df)

# # Streamlit UI
# st.set_page_config(page_title="🌿 Supply Chain Sustainability Verifier", layout="wide")
# st.title("🌿 AI-Powered Sustainable Supply Chain Verification")

# uploaded_file = st.file_uploader("Upload Supplier Data CSV", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.success("✅ Custom supplier data loaded.")
# else:
#     st.warning("📂 No file uploaded — using synthetic sample data.")
#     df = generate_sample_data()

# st.subheader("Supplier Dataset")
# st.dataframe(df)

# X_train, X_test, y_train, y_test = preprocess_data(df)
# model = train_model(X_train, y_train)

# st.subheader("Predictions on Latest Entries")
# latest_entries = df.tail(5).copy()
# input_df = latest_entries[['Carbon_Emission', 'Water_Usage', 'Labor_Practices_Score', 'Certification']]
# predictions, probabilities = predict(model, input_df)

# latest_entries['Prediction'] = predictions
# latest_entries['Confidence'] = probabilities[:, 1]
# st.dataframe(latest_entries)

# st.success("✅ Sustainability analysis complete.")


import streamlit as st
import pandas as pd
import random
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Synthetic base data to train a lightweight demo model ---
sustainability_notes = [
    "Certified ISO 14001 with carbon neutrality",
    "No sustainability report available",
    "Fair labor practices and green sourcing",
    "Environmental violations in 2023",
    "Uses renewable energy and eco packaging",
    "Waste management compliance pending",
    "High emissions and no compliance",
    "Member of ethical sourcing alliance",
    "Poor labor standards reported",
    "Full transparency and sustainability score A+"
]
labels = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  # 1: Sustainable, 0: Risky
train_df = pd.DataFrame({"notes": sustainability_notes, "label": labels})

# --- Train basic NLP model ---
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
model_pipeline.fit(train_df["notes"], train_df["label"])

# --- Streamlit App UI ---
st.set_page_config(page_title="Supply Chain Verifier", page_icon="🌱")
st.title("🌱 AI-Powered Sustainable Supply Chain Verification")
st.markdown("Upload a CSV with supplier notes or use demo data to verify sustainability risks.")

# File upload
uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])

# Load data
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Try to find a suitable text column
        text_col = None
        for col in input_df.columns:
            if "note" in col.lower() or "desc" in col.lower():
                text_col = col
                break

        if text_col is None:
            st.error("Could not find a suitable text column (e.g., 'notes', 'description').")
        else:
            input_df["Sustainability Prediction"] = model_pipeline.predict(input_df[text_col])
            input_df["Result"] = input_df["Sustainability Prediction"].map({1: "✅ Sustainable", 0: "⚠️ Risky"})
            st.success("Sustainability analysis completed.")
            st.dataframe(input_df[[text_col, "Result"]])

    except Exception as e:
        st.error(f"🚨 Error reading file: {str(e)}")

else:
    # Demo mode with fake data
    st.subheader("Demo Data Preview:")
    demo_df = pd.DataFrame({
        "supplier": [f"Supplier_{i}" for i in range(10)],
        "notes": random.choices(sustainability_notes, k=10)
    })
    demo_df["Sustainability Prediction"] = model_pipeline.predict(demo_df["notes"])
    demo_df["Result"] = demo_df["Sustainability Prediction"].map({1: "✅ Sustainable", 0: "⚠️ Risky"})
    st.dataframe(demo_df[["supplier", "notes", "Result"]])

# Footer
st.markdown("---")
st.markdown("🔍 **Ethical Guardrails:** This model emphasizes transparency, explainability, and avoids using sensitive/private data.")
