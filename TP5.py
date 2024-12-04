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
import shap
import time
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression







# #
# # =================================================Exercice 1 : =================================================
# #

print(f"======================Exercice 1 : ======================\n")



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

print(f"======================Exercice 2 : ======================\n")


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


print(f"======================Exercice 3 : ======================\n")



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



# #
# # =================================================Exercice 4 : =================================================
# #


print(f"======================Exercice 4 : ======================\n")


# # Fonction pour expliquer le modèle Random Forest avec SHAP
# def explain_rf_with_shap(X_train, y_train, rf_best_estimator):
#     # Conversion des données en vecteurs TF-IDF
#     tfidf = TfidfVectorizer(stop_words='english')
#     X_train_tfidf = tfidf.fit_transform(X_train)
#     feature_names = tfidf.get_feature_names_out()

#     # Création d'un explainer SHAP
#     explainer = shap.TreeExplainer(rf_best_estimator.named_steps['randomforest'])
    
#     # Calcul des valeurs SHAP pour un sous-ensemble des données d'entraînement
#     shap_values = explainer.shap_values(X_train_tfidf[:100].toarray())  # Pour éviter les problèmes de performance

#     # Résumé des valeurs SHAP pour la classe "Spam"
#     print("\nSHAP Summary Plot for the Random Forest Model:")
#     shap.summary_plot(shap_values[1], X_train_tfidf[:100].toarray(), feature_names=feature_names)

#     # Fonction pour obtenir les caractéristiques les plus influentes
#     def get_top_features(shap_values, feature_names, n=10):
#         avg_shap_values = np.abs(shap_values[1]).mean(axis=0)
#         top_indices = avg_shap_values.argsort()[-n:][::-1]
#         return [(feature_names[i], avg_shap_values[i]) for i in top_indices]

#     # Affichage des top caractéristiques
#     top_features = get_top_features(shap_values, feature_names)
#     print("\nTop features for Spam classification using Random Forest:")
#     for feature, importance in top_features:
#         print(f"{feature}: {importance:.4f}")

# # Réentraîner un modèle Random Forest pour l'explication SHAP
# rf_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english')),
#     ('randomforest', RandomForestClassifier(n_estimators=100, random_state=42))
# ])
# rf_pipeline.fit(X_train, y_train)

# # Appel de la fonction pour expliquer les décisions du modèle avec SHAP
# explain_rf_with_shap(X_train, y_train, rf_pipeline)





#=================================================Exercice 5 : =================================================


print(f"======================Exercice 5 : ======================\n")


# Fonction pour comparer les modèles en utilisant classification_report
def compare_models_with_classification_report(models, X_test, y_test):
    """Compare models using classification_report."""
    results = []

    for model_name, model in models.items():
        # Mesurer le temps d'entraînement
        start_train_time = time.time()
        model.fit(X_train, y_train)
        end_train_time = time.time()
        
        # Mesurer le temps de prédiction
        start_pred_time = time.time()
        y_pred = model.predict(X_test)
        end_pred_time = time.time()
        
        # Récupérer le rapport de classification
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Ajouter les résultats au tableau
        results.append({
            'Model': model_name,
            'Precision (weighted avg)': report['weighted avg']['precision'],
            'Recall (weighted avg)': report['weighted avg']['recall'],
            'F1-Score (weighted avg)': report['weighted avg']['f1-score'],
            'Training Time (s)': end_train_time - start_train_time,
            'Prediction Time (s)': end_pred_time - start_pred_time
        })

    # Affichage des résultats sous forme de tableau
    results_df = pd.DataFrame(results).sort_values(by='F1-Score (weighted avg)', ascending=False)
    print("\nComparaison des modèles:")
    print(results_df)

    return results_df

# Préparation des modèles à comparer
models = {
    'SVM Linear': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SVC(kernel='linear', probability=True))
    ]),
    'SVM RBF': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SVC(kernel='rbf', probability=True))
    ]),
    # 'Random Forest': rf_pipeline,
    'Gradient Boosting': Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    # 'HMM': model_hmm
}

# Appel de la fonction de comparaison
compare_models_with_classification_report(models, X_test, y_test)



# Création de pipelines pour les modèles de base avec transformation du texte
base_models = [
    ('gradient_boosting', Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])),
    ('svm', Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]))
]

# Définir le méta-modèle
meta_model = LogisticRegression()

# Créer le Stacking Classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Diviser les données (Assurez-vous que X_train et y_train sont définis)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Entraîner le modèle de stacking
print("Training Stacking Classifier...")
stacking_clf.fit(X_train, y_train)

# Évaluation sur l'ensemble de test
y_pred_stacking = stacking_clf.predict(X_test)

# Afficher le rapport de classification
print("\nStacking Classifier Classification Report:")
print(classification_report(y_test, y_pred_stacking))

# Matrice de confusion pour le Stacking Classifier
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Purples', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Stacking Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()




#
# =================================================Exercice 6 : =================================================
#

print(f"======================Exercice 6 : ======================\n")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam




# Charger et nettoyer les données
url = "spam.csv"
data = pd.read_csv(url, encoding='latin-1', header=None, names=['label', 'message'], usecols=[0, 1])
data['message'] = data['message'].str.replace(r'\W+', ' ', regex=True).str.lower()  # Nettoyage des messages
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Mapping des étiquettes

# Suppression des valeurs NaN s'il y en a
data = data.dropna(subset=['label'])

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.3, random_state=42
)

# Utilisation de TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Limite à 5000 caractéristiques pour réduire la dimensionnalité
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Création du modèle ANN
model = Sequential([
    Dense(128, input_shape=(X_train_tfidf.shape[1],), activation='relu'),  # Couche d'entrée + couche cachée 1
    Dropout(0.5),  # Dropout pour éviter le surapprentissage
    Dense(64, activation='relu'),  # Couche cachée 2
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Couche de sortie pour classification binaire
])

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])  # Utilisation de fonction de perte et d'optimiseur

# Entraînement
history = model.fit(
    X_train_tfidf, y_train,
    epochs=10,  # Nombre d'époques
    batch_size=32,  # Taille de lot
    validation_split=0.3,  # 30% des données d'entraînement pour la validation
    verbose=1
)

# Évaluation
loss, accuracy = model.evaluate(X_test_tfidf, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test_tfidf).round()  # On arrondit les prédictions pour les convertir en 0 ou 1

# Affichage du classification_report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Affichage de la courbe ROC-AUC
y_pred_prob = model.predict(X_test_tfidf)  # Probabilités de prédiction

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Tracé de la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Ligne de base
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC-AUC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
