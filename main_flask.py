# main_flask.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
# CORRECT:
from fpdf import FPDF
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # <<— important
import matplotlib.pyplot as plt

import seaborn as sns
import io
import base64
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "C:/Personne3/services/extracted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

target_cols = ['Diabetes', 'Obesity', 'Cancer', 'Asthma', 'Hypertension', 'Arthritis']

# ----------------------------
# Classification endpoint (single label)
# ----------------------------
@app.route("/classify", methods=["POST"])
def classify():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(file)
        df_clean = clean_and_prepare(df)

        # Prepare table for frontend
        table_columns = df_clean.columns.tolist()
        table_values = df_clean.values.tolist()

        # Pick first available target
        target_found = [col for col in target_cols if col in df_clean.columns]
        if len(target_found) == 0:
            return jsonify({"error": "No known target column found"}), 400
        target = target_found[0]

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        rf.fit(X_res, y_res)

        proba = rf.predict_proba(X)[:, 1]
        threshold = 0.3
        preds = (proba > threshold).astype(int)

        # Generate plots and stats
        plots = generate_correlation_plot(df_clean, [target])
        numeric_df = df_clean.select_dtypes(include=["number"])
        corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
        stats = corr.head(5).to_dict()
        plots_full = generate_full_plots(df_clean)

        return jsonify({
            "target": target,
            "predictions": preds.tolist(),
            "probabilities": proba.tolist(),
            "threshold": threshold,
            "plots": plots_full,
            "stats": {target: stats},
            "table": {
                "columns": table_columns,
                "values": table_values
            }
        })

    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500

def generate_correlation_plot(df, target_cols):
    plots = {}
    numeric_df = df.select_dtypes(include=['number'])

    for target in target_cols:
        if target not in numeric_df.columns:
            continue

        plt.figure(figsize=(6,4))
        corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
        sns.barplot(x=corr.index, y=corr.values)
        plt.title(f"Top correlations for {target}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots[target] = image_base64

    return plots

# --------------------------------------
# 5 & 6 — Countplots + Feature Importances
# --------------------------------------
def generate_full_plots(df):
    plots = {}

    # -----------------------------
    # A. COUNT-PLOTS POUR LES MALADIES
    # -----------------------------
    for target in target_cols:
        if target not in df.columns:
            continue

        plt.figure(figsize=(6,4))
        sns.countplot(x=df[target], color='skyblue')  # <-- replace palette='Set2'
        plt.title(f"Nombre de patients avec {target}")
        plt.tight_layout()


        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plots[f"countplot_{target}"] = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

    # -----------------------------
    # B. FEATURE IMPORTANCES POUR CHAQUE MALADIE
    # -----------------------------
    df_features = df.drop(columns=['Name'], errors='ignore')

    for target in target_cols:
        if target not in df_features.columns:
            continue

        X = df_features.drop(columns=target_cols, errors='ignore')
        y = df_features[target]

        try:
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X, y)

            feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
            sorted_feat = feat_importances.sort_values(ascending=False)

            plt.figure(figsize=(10,5))
            sorted_feat.plot(kind='bar')
            plt.title(f"Importance des features pour prédire {target}")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plots[f"importance_{target}"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

        except Exception as e:
            print(f"Feature importance error for {target}: {e}")

    return plots

# ----------------------------
# Data Mining endpoint
# ----------------------------
@app.route("/stats", methods=["POST"])
def stats():
    try:
        input_data = request.json
        df = pd.DataFrame(input_data["data"]["values"], columns=input_data["data"]["columns"])
        target_cols_input = input_data["target_cols"]

        # Keep only numeric features
        numeric_df = df.select_dtypes(include=['number'])

        stats = {}
        for target in target_cols_input:
            if target not in numeric_df.columns:
                continue
            corr = numeric_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
            stats[target] = corr.head(5).to_dict()

        # Generate correlation plots
        plots_corr = generate_correlation_plot(numeric_df, target_cols_input)
        plots_full = generate_full_plots(df)

        return jsonify({
            "stats": stats,
            "plots": plots_full
        })
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500



# ----------------------------
# PDF export helper functions
# ----------------------------
def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Name'] = df['Name'].str.title()
    df = df.drop_duplicates()
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Duration_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    le_cols = ['Doctor', 'Hospital', 'Insurance Provider', 'Medication', 'Test Results']
    for col in le_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Test Results'] = df['Test Results'].map({'Normal':0,'Abnormal':1,'Inconclusive':2})

    for cond in target_cols:
        df[cond] = df['Medical Condition'].str.contains(cond).astype(int)

    df = df.drop(['Name','Medical Condition','Date of Admission','Discharge Date'], axis=1)
    numeric_cols = ['Age','Billing Amount','Room Number','Duration_of_Stay']
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    return df

def export_to_pdf(rawCsvColumns, rawCsvData, classificationResults):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Patient Data and Classification Results", ln=True, align="C")
    pdf.ln(5)

    # Loop through rows
    for idx, row in enumerate(rawCsvData):
        pdf.set_font("Helvetica", "B", 12)
        patient_name = row[0] if len(row) > 0 else f"Patient {idx+1}"
        pdf.cell(0, 8, f"{patient_name}", ln=True)

        pdf.set_font("Helvetica", "", 10)
        for i, col in enumerate(rawCsvColumns):
            value = row[i] if i < len(row) else ""
            pdf.cell(0, 6, f"{col}: {value}", ln=True)

        # Add classifications if available
        classifications = classificationResults.get(str(idx), {})
        if classifications:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, "Classifications:", ln=True)
            for cond, status in classifications.items():
                pdf.cell(0, 6, f"{cond} - {status}", ln=True)

        pdf.ln(4)

    return pdf.output(dest='S')# return as bytes


# ----------------------------
# File upload
# ----------------------------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        file = request.files['file']
        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print(f"File received and saved to: {file_path}")
        return jsonify({"message": f"Fichier {filename} uploadé et traité avec succès!"})
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"message": f"Erreur serveur: {str(e)}"}), 500

# ----------------------------
# Download PDF from uploaded CSV
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()
    classificationResults = data.get("classifications", {})
    rawCsvColumns = data.get("columns", [])
    rawCsvData = data.get("rows", [])

    pdf_bytes = io.BytesIO(export_to_pdf(rawCsvColumns, rawCsvData, classificationResults))
    pdf_bytes.seek(0)

    return send_file(
        pdf_bytes,
        download_name="PatientDashboard.pdf",
        mimetype="application/pdf"
    )




# ----------------------------
# Multi-label model preparation
# ----------------------------
def clean_and_prepare(df):
    df = df.copy()

    # Uniformize names and remove duplicates
    df['Name'] = df['Name'].str.title()
    df = df.drop_duplicates()

    # Convert dates
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
    df['Duration_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days.fillna(0)

    # Map Gender and Test Results
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}).fillna(0)
    df['Test Results'] = df['Test Results'].map({'Normal':0, 'Abnormal':1, 'Inconclusive':2}).fillna(0)

    # Multi-label target encoding
    for cond in target_cols:
        df[cond] = df['Medical Condition'].str.contains(cond, na=False).astype(int)

    # Drop unused columns
    df = df.drop(['Name','Medical Condition','Date of Admission','Discharge Date'], axis=1, errors='ignore')

    # Encode categorical columns
    cat_cols = ['Doctor', 'Hospital', 'Insurance Provider', 'Medication']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # One-hot encode Blood Type and Admission Type
    if 'Blood Type' in df.columns:
        df = pd.get_dummies(df, columns=['Blood Type'])
    if 'Admission Type' in df.columns:
        df = pd.get_dummies(df, columns=['Admission Type'])

    # Convert numeric columns safely
    numeric_cols = ['Age','Billing Amount','Room Number','Duration_of_Stay']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Scale numeric columns
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    # Optional: create interaction feature like in your notebook
    if 'Age' in df.columns and 'Blood Type_A+' in df.columns:
        df['Age_x_BloodType_A+'] = df['Age'] * df['Blood Type_A+']

    return df




def train_multi_label_model(df):
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE for each label separately
    X_res = X_train.copy()
    y_res = y_train.copy()
    for col in target_cols:
        sm = SMOTE(random_state=42)
        X_res_col, y_res_col = sm.fit_resample(X_train, y_train[col])
        X_res = pd.DataFrame(X_res_col, columns=X_train.columns)
        y_res[col] = y_res_col

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    multi_rf = MultiOutputClassifier(rf)
    multi_rf.fit(X_res, y_res)

    y_pred = multi_rf.predict(X_test)
    report = {}
    for i, col in enumerate(target_cols):
        report[col] = classification_report(y_test[col], y_pred[:, i], output_dict=True)

    return multi_rf, report, X_test, y_test

# ----------------------------
# CSV Prediction endpoint
# ----------------------------
# Safe predict_proba handling for multi-class/single-class
def safe_predict_proba(model, X):
    proba_list = []
    for est in model.estimators_:
        p = est.predict_proba(X)
        if p.shape[1] == 1:
            # only one class present, create 2-column array
            p = np.hstack([1 - p, p])
        proba_list.append(p)
    return proba_list

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400
        file = request.files['file']
        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400

        # Read CSV
        df = pd.read_csv(file)

        # Clean and prepare data
        df_clean = clean_and_prepare(df)
        X_all = df_clean.drop(columns=target_cols)
        y_all = df_clean[target_cols]

        # Train multi-output model on all data (for demo/prediction)
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model = MultiOutputClassifier(rf)
        model.fit(X_all, y_all)

        # Make predictions
        predictions = model.predict(X_all)
        proba_list = safe_predict_proba(model, X_all)

        result = {
            "predictions": predictions.tolist(),
            "probabilities": [p[:,1].tolist() for p in proba_list],
        }
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
