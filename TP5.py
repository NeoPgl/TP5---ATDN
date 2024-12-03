#
# PAGLIARA--RIGHI Neo
# M1 OIVM




#
# =================================================Exercice 1 : =================================================
#




# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import xgboost as xgb


# 1. Charger le jeu de données
url = "spam.csv"

# Nettoyage et renommage des colonnes
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()  # Nettoyage des messages
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Mapping des étiquettes

# Suppression des valeurs NaN s'il y en a
data = data.dropna(subset=['label'])

# Vérification après nettoyage
print(f"Après nettoyage, nombre de messages : {len(data)}")
print(data.head())

# 2. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.3, random_state=42     # Divise les données en 70/30
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# 3. Entraînement et évaluation du modèle SVM non-linéaire avec noyau RBF
def non_linear_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate non-linear SVM with RBF kernel."""
    # Hyperparamètres à tester
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [0.01, 0.1, 1, 10]
    }
    
    # Pipeline avec TF-IDF et SVM
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SVC(kernel='rbf'))
    ])
    
    # Recherche des meilleurs hyperparamètres
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Meilleur modèle et prédictions
    y_pred = grid_search.predict(X_test)
    
    # Rapport de classification
    print("\nNon-Linear SVM Classification Report:")
    print(classification_report(y_test, y_pred))

    
    
     # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - SVM RBF')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()
    
    return grid_search.best_estimator_, y_pred

# Appel de la fonction SVM
best_model, y_pred = non_linear_svm(X_train, X_test, y_train, y_test)


# 4. Ajout d'un rapport pour le SVM linéaire
def linear_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate SVM with linear kernel."""
    # Pipeline avec TF-IDF et SVM linéaire
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SVC(kernel='linear'))
    ])
    
    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred_linear = pipeline.predict(X_test)
    
    # Rapport de classification
    print("\nLinear SVM Classification Report:")
    print(classification_report(y_test, y_pred_linear))
    
      # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_linear)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - SVM Linear')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

# Appel de la fonction pour SVM linéaire après le SVM non linéaire
linear_svm(X_train, X_test, y_train, y_test)



