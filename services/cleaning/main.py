# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(csv_path):
    """Charge le dataset et effectue le nettoyage de base"""
    df = pd.read_csv(csv_path)
    
    # Uniformiser les noms et supprimer doublons
    df['Name'] = df['Name'].str.title()
    df = df.drop_duplicates()
    
    # Convertir les dates et calculer la durée du séjour
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Duration_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    
    # Encodage catégoriel
    le_cols = ['Doctor', 'Hospital', 'Insurance Provider', 'Medication', 'Test Results']
    for col in le_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Encodage binaire
    df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
    df['Test Results'] = df['Test Results'].map({'Normal':0,'Abnormal':1,'Inconclusive':2})
    
    # Colonnes pour les maladies
    conditions = ['Diabetes','Obesity','Cancer','Asthma','Hypertension','Arthritis']
    for cond in conditions:
        df[cond] = df['Medical Condition'].str.contains(cond).astype(int)
    
    # Supprimer colonnes inutiles
    df = df.drop(['Name','Medical Condition','Date of Admission','Discharge Date'], axis=1)
    
    # Standardiser colonnes numériques
    numeric_cols = ['Age','Billing Amount','Room Number','Duration_of_Stay']
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    
    return df, conditions

def plot_correlation(df, conditions):
    """Affiche la heatmap de corrélation entre les maladies"""
    plt.figure(figsize=(8,6))
    corr_matrix = df[conditions].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Corrélation entre maladies")
    plt.show()

# ----------------------------
# Utilisation
# ----------------------------
csv_path = r"C:\Personne3\services\cleaning\health.csv"
df, conditions = load_and_clean_data(csv_path)
plot_correlation(df, conditions)
