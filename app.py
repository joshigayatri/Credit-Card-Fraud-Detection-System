import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(
    open("models/fraud_model.pkl", "rb")
)
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:52px;
    font-weight:bold;
    color:#1e3a8a;
    margin-top:10px;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:#555;
    margin-bottom:25px;
}

.metric-card{
    background:#EAF1FF;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.footer{
    text-align:center;
    margin-top:50px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("💳 Project Dashboard")

    st.markdown("---")

    st.write("### Project Details")

    st.write("**Domain:** FinTech")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Dataset:** Credit Card Fraud Dataset")
    st.write("**Accuracy:** 99.90%")

    st.markdown("---")

    st.write("### Technologies")

    st.write("✅ Python")
    st.write("✅ Pandas")
    st.write("✅ Scikit-Learn")
    st.write("✅ Streamlit")
    st.write("✅ Machine Learning")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">💳 Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered FinTech Security Platform</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class='metric-card'>
        Accuracy<br>
        99.90%
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class='metric-card'>
        Precision<br>
        81.82%
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class='metric-card'>
        Recall<br>
        55.10%
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")
st.write("")

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["🔍 Fraud Prediction",
     "📊 Analytics",
     "📘 About Project"]
    )

# --------------------------------------------------
# TAB 1
# --------------------------------------------------

# --------------------------------------------------
# TAB 1
# --------------------------------------------------

with tab1:

    st.subheader("🔍 Fraud Prediction")

    st.info(
        "Upload a CSV file containing transaction data "
        "(same format as the training dataset)."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        st.write("### Uploaded Data")
        st.dataframe(data.head())

        if st.button("Predict Fraud"):

            predictions = model.predict(data)

            data["Prediction"] = predictions

            data["Prediction"] = data["Prediction"].map({
                0: "Genuine",
                1: "Fraud"
            })

            csv = data.to_csv(index=False)

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

            st.write("### Prediction Results")
            st.dataframe(data)

            fraud_count = int(sum(predictions))
            genuine_count = len(predictions) - fraud_count

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "🚨 Fraud Transactions",
                    fraud_count
                )

            with col2:
                st.metric(
                    "✅ Genuine Transactions",
                    genuine_count
                )

            # Pie Chart
            import matplotlib.pyplot as plt

            st.write("### Transaction Distribution")

            fig, ax = plt.subplots()

            ax.pie(
                [genuine_count, fraud_count],
                labels=["Genuine", "Fraud"],
                autopct="%1.1f%%"
            )

            st.pyplot(fig)

            st.write("### Fraud Detection Summary")

            if fraud_count > 0:
                st.error(
                    f"🚨 {fraud_count} Fraud Transactions Detected"
                )
            else:
                st.success(
                    "✅ No Fraud Transactions Found"
                )

# --------------------------------------------------
# TAB 2
# --------------------------------------------------

with tab2:

    st.subheader("Dataset Insights")
    col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", "284,807")

with col2:
    st.metric("Fraud Cases", "492")

with col3:
    st.metric("Genuine Cases", "284,315")

    data = {
        "Category": [
            "Genuine Transactions",
            "Fraud Transactions"
        ],
        "Count": [
            284315,
            492
        ]
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.bar_chart(
        df.set_index("Category")
    )

    st.info(
        """
        Fraud transactions are very rare
        compared to genuine transactions.
        This makes fraud detection a
        challenging machine learning problem.
        """
    )
    st.subheader("Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        "99.90%",
        "99.95%"
    ]
})

st.table(comparison)

# --------------------------------------------------
# TAB 3
# --------------------------------------------------

with tab3:

    st.subheader("Project Overview")

    st.write("""
    This project uses Machine Learning
    to detect fraudulent credit card
    transactions.

    Workflow:

    1. Data Collection
    2. Data Preprocessing
    3. Machine Learning Training
    4. Fraud Prediction
    5. Risk Analysis
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - Streamlit
    - Machine Learning
    """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
"""
<hr>

<div class='footer'>

Developed By<br>

<b>Gayatri Joshi</b><br>

Final Year Engineering Project<br>

FinTech Domain

</div>
""",
unsafe_allow_html=True
)