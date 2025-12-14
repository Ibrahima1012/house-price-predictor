# 04_predict.py
# Script de prédiction sur le test set

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" " * 15 + "GÉNÉRATION DES PRÉDICTIONS")
print("=" * 70)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n[1] CHARGEMENT DES DONNÉES")
print("-" * 70)

X_test = pd.read_csv('dataset/processed/X_test.csv')
test_ids = pd.read_csv('dataset/processed/test_ids.csv')

print(f"✓ X_test : {X_test.shape}")
print(f"✓ Test IDs : {test_ids.shape}")

# ============================================
# 2. CHARGEMENT DES MODÈLES
# ============================================
print("\n[2] CHARGEMENT DES MODÈLES")
print("-" * 70)

# Charger tous les modèles
with open('models/linear_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)
print("✓ Régression Linéaire chargée")

with open('models/ridge_regression.pkl', 'rb') as f:
    ridge_model = pickle.load(f)
print("✓ Ridge Regression chargée")

with open('models/lasso_regression.pkl', 'rb') as f:
    lasso_model = pickle.load(f)
print("✓ Lasso Regression chargée")

with open('models/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)
print("✓ Random Forest chargé")

with open('models/xgboost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
print("✓ XGBoost chargé")

# ============================================
# 3. GÉNÉRATION DES PRÉDICTIONS
# ============================================
print("\n[3] GÉNÉRATION DES PRÉDICTIONS")
print("-" * 70)

# Prédictions avec chaque modèle (en échelle log)
pred_lr = lr_model.predict(X_test)
pred_ridge = ridge_model.predict(X_test)
pred_lasso = lasso_model.predict(X_test)
pred_rf = rf_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

print("✓ Prédictions générées pour tous les modèles")

# Retransformation inverse du log
pred_lr_original = np.expm1(pred_lr)
pred_ridge_original = np.expm1(pred_ridge)
pred_lasso_original = np.expm1(pred_lasso)
pred_rf_original = np.expm1(pred_rf)
pred_xgb_original = np.expm1(pred_xgb)

print("✓ Transformation inverse appliquée")

# Moyenne pondérée (ensemble) - meilleure approche
# Donner plus de poids aux meilleurs modèles
pred_ensemble = (0.1 * pred_lr + 
                 0.15 * pred_ridge + 
                 0.15 * pred_lasso + 
                 0.25 * pred_rf + 
                 0.35 * pred_xgb)
pred_ensemble_original = np.expm1(pred_ensemble)

print("✓ Prédictions ensemble créées (moyenne pondérée)")

# ============================================
# 4. STATISTIQUES DES PRÉDICTIONS
# ============================================
print("\n[4] STATISTIQUES DES PRÉDICTIONS")
print("-" * 70)

print(f"\nRégression Linéaire:")
print(f"  Prix moyen prédit : ${pred_lr_original.mean():,.2f}")
print(f"  Prix min          : ${pred_lr_original.min():,.2f}")
print(f"  Prix max          : ${pred_lr_original.max():,.2f}")

print(f"\nRidge Regression:")
print(f"  Prix moyen prédit : ${pred_ridge_original.mean():,.2f}")
print(f"  Prix min          : ${pred_ridge_original.min():,.2f}")
print(f"  Prix max          : ${pred_ridge_original.max():,.2f}")

print(f"\nLasso Regression:")
print(f"  Prix moyen prédit : ${pred_lasso_original.mean():,.2f}")
print(f"  Prix min          : ${pred_lasso_original.min():,.2f}")
print(f"  Prix max          : ${pred_lasso_original.max():,.2f}")

print(f"\nRandom Forest:")
print(f"  Prix moyen prédit : ${pred_rf_original.mean():,.2f}")
print(f"  Prix min          : ${pred_rf_original.min():,.2f}")
print(f"  Prix max          : ${pred_rf_original.max():,.2f}")

print(f"\nXGBoost (MEILLEUR):")
print(f"  Prix moyen prédit : ${pred_xgb_original.mean():,.2f}")
print(f"  Prix min          : ${pred_xgb_original.min():,.2f}")
print(f"  Prix max          : ${pred_xgb_original.max():,.2f}")

print(f"\nEnsemble (Moyenne pondérée):")
print(f"  Prix moyen prédit : ${pred_ensemble_original.mean():,.2f}")
print(f"  Prix min          : ${pred_ensemble_original.min():,.2f}")
print(f"  Prix max          : ${pred_ensemble_original.max():,.2f}")

# ============================================
# 5. CRÉATION DES FICHIERS DE SOUMISSION
# ============================================
print("\n[5] CRÉATION DES FICHIERS DE SOUMISSION")
print("-" * 70)

import os
if not os.path.exists('submissions'):
    os.makedirs('submissions')

# Soumission XGBoost (meilleur modèle)
submission_xgb = pd.DataFrame({
    'Id': test_ids['Id'],
    'SalePrice': pred_xgb_original
})
submission_xgb.to_csv('submissions/submission_xgboost.csv', index=False)
print("✓ submission_xgboost.csv")

# Soumission Random Forest
submission_rf = pd.DataFrame({
    'Id': test_ids['Id'],
    'SalePrice': pred_rf_original
})
submission_rf.to_csv('submissions/submission_random_forest.csv', index=False)
print("✓ submission_random_forest.csv")

# Soumission Ensemble
submission_ensemble = pd.DataFrame({
    'Id': test_ids['Id'],
    'SalePrice': pred_ensemble_original
})
submission_ensemble.to_csv('submissions/submission_ensemble.csv', index=False)
print("✓ submission_ensemble.csv (RECOMMANDÉ)")

# Toutes les prédictions dans un seul fichier
all_predictions = pd.DataFrame({
    'Id': test_ids['Id'],
    'LinearRegression': pred_lr_original,
    'Ridge': pred_ridge_original,
    'Lasso': pred_lasso_original,
    'RandomForest': pred_rf_original,
    'XGBoost': pred_xgb_original,
    'Ensemble': pred_ensemble_original
})
all_predictions.to_csv('submissions/all_predictions.csv', index=False)
print("✓ all_predictions.csv")

# ============================================
# 6. VISUALISATION DES PRÉDICTIONS
# ============================================
print("\n[6] VISUALISATION DES PRÉDICTIONS")
print("-" * 70)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('DISTRIBUTION DES PRÉDICTIONS PAR MODÈLE', fontsize=16, fontweight='bold')

# Linear Regression
axes[0, 0].hist(pred_lr_original, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Régression Linéaire')
axes[0, 0].set_xlabel('Prix prédit ($)')
axes[0, 0].set_ylabel('Fréquence')

# Ridge
axes[0, 1].hist(pred_ridge_original, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_title('Ridge Regression')
axes[0, 1].set_xlabel('Prix prédit ($)')

# Lasso
axes[0, 2].hist(pred_lasso_original, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 2].set_title('Lasso Regression')
axes[0, 2].set_xlabel('Prix prédit ($)')

# Random Forest
axes[1, 0].hist(pred_rf_original, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].set_title('Random Forest')
axes[1, 0].set_xlabel('Prix prédit ($)')
axes[1, 0].set_ylabel('Fréquence')

# XGBoost
axes[1, 1].hist(pred_xgb_original, bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_title('XGBoost (Meilleur)')
axes[1, 1].set_xlabel('Prix prédit ($)')

# Ensemble
axes[1, 2].hist(pred_ensemble_original, bins=50, edgecolor='black', alpha=0.7, color='teal')
axes[1, 2].set_title('Ensemble (Recommandé)')
axes[1, 2].set_xlabel('Prix prédit ($)')

plt.tight_layout()
plt.savefig('submissions/predictions_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique sauvegardé : predictions_distribution.png")

# Comparaison des prédictions
plt.figure(figsize=(12, 6))
plt.plot(pred_lr_original[:100], label='Linear Regression', alpha=0.6)
plt.plot(pred_ridge_original[:100], label='Ridge', alpha=0.6)
plt.plot(pred_lasso_original[:100], label='Lasso', alpha=0.6)
plt.plot(pred_rf_original[:100], label='Random Forest', alpha=0.6)
plt.plot(pred_xgb_original[:100], label='XGBoost', alpha=0.8, linewidth=2)
plt.plot(pred_ensemble_original[:100], label='Ensemble', alpha=0.8, linewidth=2, 
         linestyle='--', color='black')
plt.xlabel('Index du test')
plt.ylabel('Prix prédit ($)')
plt.title('Comparaison des prédictions - 100 premières maisons', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('submissions/predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique sauvegardé : predictions_comparison.png")

print("\n" + "=" * 70)
print(" " * 15 + "✓ PRÉDICTIONS TERMINÉES !")
print("=" * 70)
print(f"\n📁 Fichiers de soumission créés dans : submissions/")
print(f"📊 Utilisez 'submission_ensemble.csv' pour la meilleure performance")
print("\nProchaine étape : Application web interactive (05_app.py)")