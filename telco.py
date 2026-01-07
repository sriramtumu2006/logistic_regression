import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config("Customer Churn Prediction", layout="centered")
def load_css(filename):
    with open(filename) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h1>Customer Churn Prediction</h1>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

df.drop("customerID", axis=1, inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="Churn", data=df, ax=ax1)
ax1.set_xticklabels(["No Churn", "Churn"])
st.pyplot(fig1)
st.markdown('</div>', unsafe_allow_html=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2f}")
c2.metric("Misclassification Rate", f"{1 - accuracy:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig2, ax2 = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)
disp.plot(cmap=plt.cm.Greens, ax=ax2)
st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.markdown('</div>', unsafe_allow_html=True)

tn, fp, fn, tp = cm.ravel()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Business Insights")

st.markdown(f"""
✅ **Churn customers correctly identified (True Positives – TP):** {tp}  

❌ **Non-churn customers misclassified as churn (False Positives – FP):** {fp}  

⚠️ **Missed churn customers (False Negatives – FN):** {fn}  

📉 **Correctly identified loyal customers (True Negatives – TN):** {tn}  
""")

st.markdown('</div>', unsafe_allow_html=True)
