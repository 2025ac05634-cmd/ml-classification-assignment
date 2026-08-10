"""
Streamlit Web Application - Credit Card Fraud Detection
Interactive ML application for detecting fraudulent transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff0000;
    }
    .legitimate-card {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00aa00;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection System")
st.markdown("### Machine Learning Models for Detecting Fraudulent Transactions")
st.write("Analyze transactions and predict fraud risk using 5 different ML models")


# ============================================
# LOAD TRAINED MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing utilities"""
    models = {}
    try:
        # Model file mappings
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'KNN': 'knn.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl'
        }

        # Load all models
        for model_name, filename in model_files.items():
            filepath = f'models/{filename}'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
            else:
                st.warning(f"Model file not found: {filepath}")

        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        return models, scaler
    except Exception as e:
        return None, None


# Load models
models, scaler = load_models()

# Check if models loaded successfully
if models is None or len(models) == 0:
    st.error("❌ Models not found! Please ensure the following files exist:")
    st.error("- models/logistic_regression.pkl")
    st.error("- models/decision_tree.pkl")
    st.error("- models/knn.pkl")
    st.error("- models/naive_bayes.pkl")
    st.error("- models/random_forest.pkl")
    st.error("- models/scaler.pkl")
    st.error("\nRun 'python train_model_CREDITCARD.py' first to generate these files.")
    st.stop()

# ============================================
# SIDEBAR - INFORMATION & CONFIGURATION
# ============================================
with st.sidebar:
    st.header("📁 Data Upload")
    st.write("Upload your credit card transactions dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your test data in CSV format (creditcard.csv or similar)"
    )

    st.markdown("---")
    st.header("🤖 Models Available")
    st.write("5 Classification Models:")
    for i, model_name in enumerate(models.keys(), 1):
        st.write(f"{i}. ✓ {model_name}")

    st.markdown("---")
    st.header("ℹ️ About This System")
    st.markdown("""
    ### Dataset Features:
    - **V1 to V28**: PCA-transformed features
    - **Amount**: Transaction amount
    - **Time**: Seconds from first transaction
    - **Class**: 0=Legitimate, 1=Fraud

    ### Why These Models:
    1. **Logistic Regression** - Fast, interpretable baseline
    2. **Decision Tree** - Handles non-linear patterns
    3. **KNN** - Finds similar transactions
    4. **Naive Bayes** - Probabilistic approach
    5. **Random Forest** - Ensemble, often best

    ### Key Metric Focus:
    - **Recall** - Critical! (catch fraud)
    - **Precision** - Avoid false alarms
    - **AUC** - Overall discrimination
    """)

# ============================================
# MAIN CONTENT - CONDITIONAL RENDERING
# ============================================
if uploaded_file is not None:
    # Load the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Display success message and statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Transactions", f"{df.shape[0]:,}")
    with col2:
        st.metric("📈 Features", f"{df.shape[1]}")
    with col3:
        if 'Class' in df.columns:
            fraud_count = (df['Class'] == 1).sum()
            st.metric("⚠️ Fraud Cases", f"{fraud_count:,}")
        else:
            st.metric("⚠️ Fraud Cases", "N/A")
    with col4:
        if 'Class' in df.columns:
            fraud_pct = (df['Class'] == 1).sum() / len(df) * 100
            st.metric("% Fraudulent", f"{fraud_pct:.2f}%")

    st.success(f"✓ Data uploaded successfully!")

    # ============================================
    # TABS - DIFFERENT VIEWS
    # ============================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["🔍 Explore", "🎯 Predictions", "📊 Model Comparison",
         "📋 Feature Info", "📈 Analysis", "ℹ️ About"]
    )

    # ============================================
    # TAB 1: DATA EXPLORATION
    # ============================================
    with tab1:
        st.header("Dataset Exploration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("📊 Dataset Info")
            info_text = f"""
            **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns

            **Memory Usage:** {df.memory_usage().sum() / 1024 ** 2:.2f} MB

            **Data Types:**
            - Numeric: {df.select_dtypes(include=[np.number]).shape[1]}
            - Categorical: {df.select_dtypes(include=['object']).shape[1]}
            """
            st.write(info_text)

        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        # Check for missing values
        st.subheader("❌ Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✓ No missing values found!")
        else:
            st.dataframe(missing[missing > 0], use_container_width=True)

        # Class distribution if available
        if 'Class' in df.columns:
            st.subheader("🎯 Class Distribution")
            class_counts = df['Class'].value_counts()

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71', '#e74c3c']
            labels = ['Legitimate (0)', 'Fraud (1)']
            bars = ax.bar(labels, [class_counts[0], class_counts.get(1, 0)], color=colors, edgecolor='black',
                          linewidth=1.5)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Transaction Classification Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    # ============================================
    # TAB 2: PREDICTIONS
    # ============================================
    with tab2:
        st.header("🎯 Make Fraud Predictions")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_model = st.selectbox(
                "Select a Classification Model:",
                list(models.keys()),
                index=4,  # Default to Random Forest
                help="Choose which model to use for fraud detection"
            )

        with col2:
            st.write("")
            st.write("")
            run_button = st.button("🚀 Run Predictions", use_container_width=True)

        if run_button:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Prepare data
                status_text.text("Step 1/4: Preparing data...")
                progress_bar.progress(25)

                X_data = df.copy()

                # Remove Time and Class if present
                if 'Time' in X_data.columns:
                    X_data = X_data.drop('Time', axis=1)
                if 'Class' in X_data.columns:
                    X_data = X_data.drop('Class', axis=1)

                # Step 2: Check features match
                status_text.text("Step 2/4: Validating features...")
                progress_bar.progress(50)

                expected_features = scaler.get_feature_names_out().tolist() if hasattr(scaler,
                                                                                       'get_feature_names_out') else None

                # Step 3: Scale features
                status_text.text("Step 3/4: Scaling features...")
                progress_bar.progress(75)

                X_scaled = scaler.transform(X_data)
                X_scaled = pd.DataFrame(X_scaled, columns=X_data.columns)

                # Step 4: Make predictions
                status_text.text("Step 4/4: Making predictions...")
                progress_bar.progress(90)

                model = models[selected_model]
                predictions = model.predict(X_scaled)

                try:
                    probabilities = model.predict_proba(X_scaled)
                except:
                    probabilities = None

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                # Display results
                st.success(f"✅ Predictions completed using {selected_model}!")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Transactions",
                        len(predictions),
                        help="Total predictions made"
                    )

                with col2:
                    fraud_pred = (predictions == 1).sum()
                    st.metric(
                        "Fraud Predictions",
                        fraud_pred,
                        help="Transactions flagged as fraud"
                    )

                with col3:
                    legit_pred = (predictions == 0).sum()
                    st.metric(
                        "Legitimate Predictions",
                        legit_pred,
                        help="Transactions flagged as legitimate"
                    )

                with col4:
                    if probabilities is not None:
                        avg_fraud_prob = np.mean(probabilities[:, 1])
                        st.metric(
                            "Avg Fraud Probability",
                            f"{avg_fraud_prob:.2%}",
                            help="Average fraud risk score"
                        )

                # Display predictions table
                st.subheader("📊 Detailed Predictions")

                results_df = pd.DataFrame({
                    'Transaction_ID': range(len(predictions)),
                    'Prediction': ['FRAUD 🚨' if p == 1 else 'LEGITIMATE ✅' for p in predictions],
                    'Class_Code': predictions
                })

                if probabilities is not None:
                    results_df['Fraud_Probability'] = probabilities[:, 1].round(4)
                    results_df['Legitimate_Probability'] = probabilities[:, 0].round(4)


                # Apply color coding
                def highlight_fraud(val):
                    if 'FRAUD' in str(val):
                        return 'background-color: #ffcccc; font-weight: bold; color: red;'
                    else:
                        return 'background-color: #ccffcc; font-weight: bold; color: green;'


                styled_df = results_df.style.applymap(
                    highlight_fraud,
                    subset=['Prediction']
                )

                st.dataframe(styled_df, use_container_width=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"{selected_model.replace(' ', '_')}_fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Fraud risk visualization
                st.subheader("📈 Fraud Risk Distribution")

                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart of predictions
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fraud_count = (predictions == 1).sum()
                    legit_count = (predictions == 0).sum()
                    colors = ['#2ecc71', '#e74c3c']
                    sizes = [legit_count, fraud_count]
                    labels = [f'Legitimate\n({legit_count})', f'Fraud 🚨\n({fraud_count})']
                    explode = (0, 0.1) if fraud_count > 0 else (0, 0)

                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           explode=explode, shadow=True, startangle=90)
                    ax.set_title(f'Fraud Detection Results - {selected_model}', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    # Probability distribution
                    if probabilities is not None:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fraud_probs = probabilities[:, 1]
                        ax.hist(fraud_probs, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
                        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
                        ax.set_xlabel('Fraud Probability', fontsize=12, fontweight='bold')
                        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                        ax.set_title('Fraud Probability Distribution', fontsize=12, fontweight='bold')
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Make sure your CSV has the same columns as the training data (V1-V28, Amount, Time)")

    # ============================================
    # TAB 3: MODEL COMPARISON
    # ============================================
    with tab3:
        st.header("📊 Model Performance Comparison")

        if os.path.exists('model_results.csv'):
            results_df = pd.read_csv('model_results.csv')

            st.subheader("📋 Metrics Comparison Table")


            # Color code the table
            def color_high(val):
                if isinstance(val, float):
                    return f'background-color: lightgreen' if val > 0.95 else ''
                return ''


            styled_results = results_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'MCC': '{:.4f}'
            })

            st.dataframe(styled_results, use_container_width=True)

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Accuracy Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(results_df['Model'], results_df['Accuracy'],
                              color='#1f77b4', edgecolor='black', linewidth=1.5)
                ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.0)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("🚨 Recall Comparison (Catch Fraud)")
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(results_df['Model'], results_df['Recall'],
                              color='#ff7f0e', edgecolor='black', linewidth=1.5)
                ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.0)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            # Precision vs Recall trade-off
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⚖️ Precision Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(results_df['Model'], results_df['Precision'],
                       color='#2ca02c', edgecolor='black', linewidth=1.5)
                ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.0)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.subheader("📊 F1 Score Comparison")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(results_df['Model'], results_df['F1 Score'],
                       color='#9467bd', edgecolor='black', linewidth=1.5)
                ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 1.0)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

            # Best model highlight
            best_idx = results_df['Accuracy'].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_acc = results_df.loc[best_idx, 'Accuracy']
            best_recall = results_df.loc[best_idx, 'Recall']
            best_f1 = results_df.loc[best_idx, 'F1 Score']

            st.markdown("---")
            st.markdown(f"""
            ### 🏆 Best Performing Model

            **Model:** {best_model}

            **Accuracy:** {best_acc:.4f} | **Recall:** {best_recall:.4f} | **F1 Score:** {best_f1:.4f}
            """)
        else:
            st.warning("⚠️ Model results not found. Please train models first!")

    # ============================================
    # TAB 4: FEATURE INFORMATION
    # ============================================
    with tab4:
        st.header("🔧 Feature Information")

        # Get numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Class' in numeric_features:
            numeric_features.remove('Class')

        st.subheader("📊 V1-V28 Features (PCA Transformed)")
        st.write(f"Total Features: {len(numeric_features)}")
        st.write("These are Principal Component Analysis (PCA) transformed features")

        # Feature correlation
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribution of Amount")
            fig, ax = plt.subplots(figsize=(8, 5))
            if 'Amount' in df.columns:
                ax.hist(df['Amount'], bins=50, edgecolor='black', color='#3498db', alpha=0.7)
                ax.set_xlabel('Transaction Amount', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Transaction Statistics")
            if 'Amount' in df.columns:
                stats_text = f"""
                **Amount Statistics:**
                - Min: ${df['Amount'].min():.2f}
                - Max: ${df['Amount'].max():.2f}
                - Mean: ${df['Amount'].mean():.2f}
                - Median: ${df['Amount'].median():.2f}
                - Std Dev: ${df['Amount'].std():.2f}
                """
                st.write(stats_text)

    # ============================================
    # TAB 5: ANALYSIS
    # ============================================
    with tab5:
        st.header("📈 Advanced Analysis")

        st.subheader("📚 About the Dataset")
        st.markdown("""
        ### Credit Card Fraud Detection Dataset

        **Dataset Characteristics:**
        - 284,807 transactions
        - 492 fraudulent transactions (0.17%)
        - Highly imbalanced dataset
        - Features V1-V28 are PCA-transformed for confidentiality

        **Challenge:** 
        Low fraud rate makes accuracy misleading. A model predicting 
        "all legitimate" would have 99.8% accuracy but catch 0% fraud!

        **Solution:** 
        Focus on **RECALL** (catch fraud) and **PRECISION** (minimize false alarms)
        rather than raw accuracy.

        ### Model Selection Guide:

        1. **Logistic Regression**
           - Fast, interpretable
           - Good baseline for comparison

        2. **Decision Tree**
           - Handles non-linear patterns
           - Prone to overfitting

        3. **K-Nearest Neighbors**
           - Finds similar transactions
           - Computationally expensive on large datasets

        4. **Naive Bayes**
           - Fast probabilistic model
           - Assumes feature independence

        5. **Random Forest** ⭐
           - Ensemble of trees
           - Usually best overall performance
           - Less prone to overfitting
        """)

    # ============================================
    # TAB 6: ABOUT
    # ============================================
    with tab6:
        st.header("ℹ️ About This Application")

        st.markdown("""
        ## 🎯 Application Overview

        This is a **Credit Card Fraud Detection System** built with machine learning.

        ### 📊 Key Features:

        1. **Multiple Models** - Compare 5 different classification algorithms
        2. **Real-time Predictions** - Classify new transactions instantly
        3. **Performance Metrics** - Track model accuracy, precision, recall, F1, AUC, MCC
        4. **Visual Analytics** - Charts and graphs for easy interpretation
        5. **Download Results** - Export predictions as CSV

        ### 🤖 Classification Models:

        1. **Logistic Regression** - Linear baseline model
        2. **Decision Tree** - Tree-based decisions
        3. **K-Nearest Neighbors** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of trees (recommended)

        ### 📈 Evaluation Metrics Explained:

        - **Accuracy**: % correct predictions (misleading for imbalanced data!)
        - **AUC Score**: Area under ROC curve (0-1, higher is better)
        - **Precision**: Of fraud predictions, % that are actually fraud
        - **Recall**: Of actual fraud, % that model catches (MOST IMPORTANT!)
        - **F1 Score**: Harmonic mean of precision and recall
        - **MCC**: Matthews Correlation Coefficient (for imbalanced data)

        ### ⚠️ Important for Fraud Detection:

        **Recall is Critical!**
        - Missing fraud (False Negative) is very costly
        - False alarms (False Positive) are acceptable

        **Precision Matters Too!**
        - Too many false positives frustrate customers
        - Need to balance recall and precision

        ### 🚀 How to Use:

        1. **Upload Data** - Click "Choose a CSV file" in sidebar
        2. **Explore** - View dataset statistics and distributions
        3. **Predict** - Select model and make predictions
        4. **Compare** - View all models' performance
        5. **Download** - Save predictions for further analysis

        ### 📚 Dataset Information:

        - **Source**: Kaggle Credit Card Fraud Detection
        - **Transactions**: 284,807 records
        - **Fraud Cases**: 492 (0.17%)
        - **Features**: V1-V28 (PCA), Amount, Time
        - **Target**: Class (0=Legitimate, 1=Fraud)

        ### 💡 Tips:

        - Check **Recall** metric to see fraud catch rate
        - Look at **Precision** to minimize false alarms  
        - Use **F1 Score** for balanced view
        - Consider **AUC** for overall model quality

        ---

        **Created with ❤️ using Streamlit and Scikit-learn**
        """)

else:
    # No file uploaded - show welcome screen
    st.info("👈 Upload a CSV file in the sidebar to get started!")

    st.markdown("""
    ## 🚀 Welcome to Credit Card Fraud Detection!

    ### Quick Start Guide

    1. **Prepare Your Data**
       - CSV file with credit card transaction data
       - Should contain features V1-V28, Amount, Time
       - Optional: Class column (0=Legitimate, 1=Fraud)

    2. **Upload File**
       - Click "Choose a CSV file" in the sidebar
       - Select your creditcard.csv file

    3. **Explore Data**
       - View statistics and distributions
       - Check class balance and missing values

    4. **Make Predictions**
       - Select any of 5 ML models
       - Get instant fraud/legitimate predictions
       - Download results as CSV

    5. **Compare Models**
       - View performance metrics for all 5 models
       - Check accuracy, precision, recall, F1, AUC, MCC
       - Identify best model for your use case

    ---

    ## 📊 Expected CSV Format

    Your CSV should contain:
    - **V1 to V28**: PCA-transformed features
    - **Amount**: Transaction amount in dollars
    - **Time**: Seconds from first transaction (optional)
    - **Class**: 0=Legitimate, 1=Fraud (optional for prediction)

    ### Sample Format:
    ```
    Time,V1,V2,V3,...,V28,Amount,Class
    0,-1.359,-0.0727,...,0.0986,149.62,0
    0,1.192,0.2661,...,0.0851,2.69,0
    1,-1.358,-1.3401,...,0.2476,378.66,1
    ```

    ---

    ## 🤖 Available Models

    1. **Logistic Regression** - Fast, interpretable
    2. **Decision Tree** - Non-linear patterns
    3. **K-Nearest Neighbors** - Similar transaction search
    4. **Naive Bayes** - Probabilistic approach
    5. **Random Forest** - Ensemble (recommended ⭐)

    ---

    ## ⚠️ Class Imbalance

    Fraud is RARE (0.17% of transactions):
    - Accuracy alone is not a good metric
    - Focus on **RECALL** (catch fraud!)
    - Balance with **PRECISION** (minimize false alarms)
    - Use **F1 Score** for overall performance

    ---

    **Upload your data to get started!** 🚀
    """)
