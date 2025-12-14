# 03_train_models.py
# Script d'entraînement et comparaison de modèles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" " * 15 + "ENTRAÎNEMENT DES MODÈLES")
print("=" * 70)

# ============================================
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ============================================
print("\n[1] CHARGEMENT DES DONNÉES PRÉTRAITÉES")
print("-" * 70)

X_train = pd.read_csv('dataset/processed/X_train.csv')
X_val = pd.read_csv('dataset/processed/X_val.csv')
y_train = pd.read_csv('dataset/processed/y_train.csv').values.ravel()
y_val = pd.read_csv('dataset/processed/y_val.csv').values.ravel()

print(f"✓ X_train : {X_train.shape}")
print(f"✓ X_val   : {X_val.shape}")
print(f"✓ y_train : {y_train.shape}")
print(f"✓ y_val   : {y_val.shape}")

# ============================================
# 2. FONCTION D'ÉVALUATION
# ============================================
def evaluate_model(y_true, y_pred, model_name):
    """Évalue un modèle avec plusieurs métriques"""
    # Retransformation inverse du log
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)
    
    # Métriques
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_original = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
    
    print(f"\n{'='*70}")
    print(f"  RÉSULTATS : {model_name}")
    print(f"{'='*70}")
    print(f"  RMSE (log)                : {rmse:.4f}")
    print(f"  RMSE (prix réel)          : ${rmse_original:,.2f}")
    print(f"  MAE (prix réel)           : ${mae:,.2f}")
    print(f"  R² Score                  : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAPE (erreur %)           : {mape:.2f}%")
    print(f"{'='*70}")
    
    return {
        'model': model_name,
        'rmse': rmse,
        'rmse_original': rmse_original,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

# ============================================
# 3. MODÈLE 1 : RÉGRESSION LINÉAIRE
# ============================================
print("\n[2] ENTRAÎNEMENT : RÉGRESSION LINÉAIRE")
print("-" * 70)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)

results_lr = evaluate_model(y_val, y_pred_lr, "Régression Linéaire")

# ============================================
# 4. MODÈLE 2 : RIDGE REGRESSION
# ============================================
print("\n[3] ENTRAÎNEMENT : RIDGE REGRESSION (avec régularisation)")
print("-" * 70)

ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_val)

results_ridge = evaluate_model(y_val, y_pred_ridge, "Ridge Regression")

# ============================================
# 5. MODÈLE 3 : LASSO REGRESSION
# ============================================
print("\n[4] ENTRAÎNEMENT : LASSO REGRESSION (sélection de features)")
print("-" * 70)

lasso_model = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_val)

results_lasso = evaluate_model(y_val, y_pred_lasso, "Lasso Regression")

# ============================================
# 6. MODÈLE 4 : RANDOM FOREST
# ============================================
print("\n[5] ENTRAÎNEMENT : RANDOM FOREST")
print("-" * 70)
print("⏳ Entraînement en cours (cela peut prendre 1-2 minutes)...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)

results_rf = evaluate_model(y_val, y_pred_rf, "Random Forest")

# ============================================
# 7. MODÈLE 5 : XGBOOST (LE MEILLEUR!)
# ============================================
print("\n[6] ENTRAÎNEMENT : XGBOOST")
print("-" * 70)
print("⏳ Entraînement en cours (cela peut prendre 1-2 minutes)...")

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)

results_xgb = evaluate_model(y_val, y_pred_xgb, "XGBoost")

# ============================================
# 8. COMPARAISON DES MODÈLES
# ============================================
print("\n[7] COMPARAISON DES MODÈLES")
print("-" * 70)

# Tableau comparatif
results_df = pd.DataFrame([results_lr, results_ridge, results_lasso, results_rf, results_xgb])
results_df = results_df.sort_values('rmse')

print("\n📊 TABLEAU COMPARATIF (trié par RMSE)")
print("="*70)
print(results_df.to_string(index=False))

# Meilleur modèle
best_model_name = results_df.iloc[0]['model']
best_rmse = results_df.iloc[0]['rmse']
best_r2 = results_df.iloc[0]['r2']

print(f"\n{'='*70}")
print(f"🏆 MEILLEUR MODÈLE : {best_model_name}")
print(f"   RMSE : {best_rmse:.4f}")
print(f"   R²   : {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"{'='*70}")

# ============================================
# 9. VISUALISATIONS
# ============================================
print("\n[8] GÉNÉRATION DES VISUALISATIONS")
print("-" * 70)

# Graphique 1 : Comparaison des RMSE
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('COMPARAISON DES MODÈLES', fontsize=16, fontweight='bold')

# RMSE Comparison
axes[0, 0].barh(results_df['model'], results_df['rmse'], color='steelblue')
axes[0, 0].set_xlabel('RMSE (log scale)')
axes[0, 0].set_title('RMSE - Plus bas = meilleur')
axes[0, 0].invert_yaxis()

# R² Score Comparison
axes[0, 1].barh(results_df['model'], results_df['r2'], color='green')
axes[0, 1].set_xlabel('R² Score')
axes[0, 1].set_title('R² Score - Plus haut = meilleur')
axes[0, 1].invert_yaxis()

# MAE Comparison
axes[1, 0].barh(results_df['model'], results_df['mae'], color='coral')
axes[1, 0].set_xlabel('MAE ($)')
axes[1, 0].set_title('Mean Absolute Error - Plus bas = meilleur')
axes[1, 0].invert_yaxis()

# MAPE Comparison
axes[1, 1].barh(results_df['model'], results_df['mape'], color='purple')
axes[1, 1].set_xlabel('MAPE (%)')
axes[1, 1].set_title('Mean Absolute Percentage Error - Plus bas = meilleur')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('dataset/processed/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique de comparaison sauvegardé : model_comparison.png")

# Graphique 2 : Prédictions vs Valeurs réelles (XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('QUALITÉ DES PRÉDICTIONS - XGBOOST', fontsize=16, fontweight='bold')

# Prix en échelle log
axes[0].scatter(y_val, y_pred_xgb, alpha=0.5, s=30)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
             'r--', lw=2, label='Prédiction parfaite')
axes[0].set_xlabel('Prix réel (log)')
axes[0].set_ylabel('Prix prédit (log)')
axes[0].set_title('Prédictions vs Réalité (échelle log)')
axes[0].legend()

# Prix en dollars
y_val_original = np.expm1(y_val)
y_pred_xgb_original = np.expm1(y_pred_xgb)
axes[1].scatter(y_val_original, y_pred_xgb_original, alpha=0.5, s=30, color='green')
axes[1].plot([y_val_original.min(), y_val_original.max()], 
             [y_val_original.min(), y_val_original.max()], 
             'r--', lw=2, label='Prédiction parfaite')
axes[1].set_xlabel('Prix réel ($)')
axes[1].set_ylabel('Prix prédit ($)')
axes[1].set_title('Prédictions vs Réalité (dollars)')
axes[1].legend()

plt.tight_layout()
plt.savefig('dataset/processed/predictions_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique des prédictions sauvegardé : predictions_xgboost.png")

# Graphique 3 : Feature Importance (XGBoost)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'], color='teal')
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('TOP 20 DES FEATURES LES PLUS IMPORTANTES (XGBoost)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('dataset/processed/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Graphique d'importance des features sauvegardé : feature_importance.png")

# ============================================
# 10. SAUVEGARDE DES MODÈLES
# ============================================
print("\n[9] SAUVEGARDE DES MODÈLES")
print("-" * 70)

# Créer dossier models
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Sauvegarde de tous les modèles
with open('models/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✓ linear_regression.pkl")

with open('models/ridge_regression.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
print("✓ ridge_regression.pkl")

with open('models/lasso_regression.pkl', 'wb') as f:
    pickle.dump(lasso_model, f)
print("✓ lasso_regression.pkl")

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✓ random_forest.pkl")

with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✓ xgboost.pkl (MEILLEUR MODÈLE)")

# Sauvegarde des résultats
results_df.to_csv('models/model_comparison.csv', index=False)
print("✓ model_comparison.csv")

print("\n" + "=" * 70)
print(" " * 15 + "✓ ENTRAÎNEMENT TERMINÉ !")
print("=" * 70)
print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"📊 Précision : {best_r2*100:.2f}%")
print(f"💰 Erreur moyenne : ${results_df.iloc[0]['mae']:,.2f}")
print("\nProchaine étape : Prédictions sur le test set (04_predict.py)")