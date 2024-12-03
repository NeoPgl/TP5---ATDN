#
# PAGLIARA--RIGHI Neo
# M1 OIVM


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
from sklearn.feature_extraction.text import CountVectorizer
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder





# #
# # =================================================Exercice 1 : =================================================
# #




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

    
    
    # Matrice de confusion - SVM NON LINEAIRE
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
    
    # Matrice de confusion - SVM LINEAIRE
    cm = confusion_matrix(y_test, y_pred_linear)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - SVM Linear')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

# Appel de la fonction pour SVM linéaire après le SVM non linéaire
linear_svm(X_train, X_test, y_train, y_test)




#
# =================================================Exercice 2 : =================================================
#

def ensemble_models(X_train, X_test, y_train, y_test):
    # Utilisation du TfidfVectorizer pour transformer le texte en vecteurs numériques
    tfidf = TfidfVectorizer(stop_words='english')

    # Pipeline pour RandomForest avec TF-IDF
    rf_params = {
        'randomforest__n_estimators': [100, 200],
        'randomforest__max_depth': [None, 10, 20],
        'randomforest__min_samples_split': [2, 5]
    }

    rf_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('randomforest', RandomForestClassifier())
    ])

    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5)
    rf_grid.fit(X_train, y_train)

    print(f"Meilleurs paramètres pour Random Forest: {rf_grid.best_params_}")

    # Pipeline pour XGBoost avec TF-IDF
    xgb_params = {
        'xgbclassifier__n_estimators': [100, 200],
        'xgbclassifier__learning_rate': [0.01, 0.1],
        'xgbclassifier__max_depth': [3, 5]
    }

    xgb_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('xgbclassifier', xgb.XGBClassifier())
    ])

    xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5)
    xgb_grid.fit(X_train, y_train)

    print(f"Meilleurs paramètres pour XGBoost: {xgb_grid.best_params_}")

    # Affichage des rapports de classification
    print("\nRandom Forest Classification Report:")
    y_pred_rf = rf_grid.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    print("\nXGBoost Classification Report:")
    y_pred_xgb = xgb_grid.predict(X_test)
    print(classification_report(y_test, y_pred_xgb))

    # Courbe ROC pour les deux modèles
    def plot_roc_curve(model, X_test, y_test, title):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.2f})')

    plt.figure(figsize=(10, 6))
    plot_roc_curve(rf_grid.best_estimator_, X_test, y_test, 'Random Forest')
    plot_roc_curve(xgb_grid.best_estimator_, X_test, y_test, 'XGBoost')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

    return rf_grid.best_estimator_, xgb_grid.best_estimator_

# Appel de la fonction avec les ensembles d'entraînement et de test
ensemble_models(X_train, X_test, y_train, y_test)




#
# =================================================Exercice 3 : =================================================
#



# Séparer les features et les labels
X = data[data.columns[1]]  # Texte
y = data[data.columns[0]]  # Labels (ham, spam)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilisation des n-grammes pour la vectorisation
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')  # Utilisation des bigrammes
X_train_ngram = vectorizer.fit_transform(X_train)
X_test_ngram = vectorizer.transform(X_test)

# Si X_train_ngram est une matrice sparse, convertissez-la en une matrice dense
X_train_dense = X_train_ngram.toarray()
X_test_dense = X_test_ngram.toarray()

# Encoder les labels "ham" et "spam"
label_encoder = LabelEncoder()
y_train_hmm = label_encoder.fit_transform(y_train)
y_test_hmm = label_encoder.transform(y_test)

# Définir le modèle HMM avec 2 états (ham et spam)
model_hmm = hmm.MultinomialHMM(n_components=2, random_state=42)

# Entraîner le modèle HMM
lengths = [len(X_train)]  # Si toutes les données sont considérées comme une seule séquence
model_hmm.fit(X_train_dense, lengths=lengths)

# Prédiction des labels sur les données de test
y_pred_hmm = model_hmm.predict(X_test_dense)

# Décodage des résultats
y_pred_hmm_decoded = label_encoder.inverse_transform(y_pred_hmm)

# Rapport de classification
print("\nHMM Classification Report:")
print(classification_report(y_test, y_pred_hmm_decoded))