# 02_preprocessing.py
# Script de prétraitement des données immobilières

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" " * 15 + "PRÉTRAITEMENT DES DONNÉES")
print("=" * 70)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n[1] CHARGEMENT DES DONNÉES")
print("-" * 70)

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"✓ Train : {train_df.shape}")
print(f"✓ Test  : {test_df.shape}")

# Sauvegarde des IDs du test set
test_ids = test_df['Id'].copy()

# Extraction de la variable cible
y_train = train_df['SalePrice'].copy()
train_df = train_df.drop(['SalePrice'], axis=1)

print(f"✓ Variable cible extraite : {y_train.shape}")

# Combinaison train + test pour preprocessing uniforme
all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
print(f"✓ Dataset combiné : {all_data.shape}")

# ============================================
# 2. GESTION DES VALEURS MANQUANTES
# ============================================
print("\n[2] GESTION DES VALEURS MANQUANTES")
print("-" * 70)

valeurs_manquantes_avant = all_data.isnull().sum().sum()
print(f"Valeurs manquantes avant traitement : {valeurs_manquantes_avant}")

# Variables catégorielles où NA = "None" (pas de cette caractéristique)
categorical_na_none = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

for col in categorical_na_none:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

print(f"✓ {len(categorical_na_none)} colonnes catégorielles : NA → 'None'")

# Variables numériques où NA = 0 (pas de cette caractéristique)
numerical_na_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

for col in numerical_na_zero:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

print(f"✓ {len(numerical_na_zero)} colonnes numériques : NA → 0")

# LotFrontage : médiane par quartier
if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"✓ LotFrontage : rempli par médiane du quartier")

# MSZoning : mode (valeur la plus fréquente)
if 'MSZoning' in all_data.columns:
    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Utilities : mode
if 'Utilities' in all_data.columns:
    all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])

# Functional : mode
if 'Functional' in all_data.columns:
    all_data['Functional'] = all_data['Functional'].fillna('Typ')

# Electrical : mode
if 'Electrical' in all_data.columns:
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenQual : mode
if 'KitchenQual' in all_data.columns:
    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# Exterior1st et Exterior2nd : mode
if 'Exterior1st' in all_data.columns:
    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
if 'Exterior2nd' in all_data.columns:
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# SaleType : mode
if 'SaleType' in all_data.columns:
    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# Traitement des dernières valeurs manquantes
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

valeurs_manquantes_apres = all_data.isnull().sum().sum()
print(f"\n✓ Valeurs manquantes après traitement : {valeurs_manquantes_apres}")

# ============================================
# 3. CRÉATION DE NOUVELLES FEATURES
# ============================================
print("\n[3] CRÉATION DE NOUVELLES FEATURES (FEATURE ENGINEERING)")
print("-" * 70)

# Surface totale de la maison
all_data['TotalSF'] = (all_data['TotalBsmtSF'] + 
                       all_data['1stFlrSF'] + 
                       all_data['2ndFlrSF'])
print("✓ TotalSF créé (surface totale)")

# Nombre total de salles de bain
all_data['TotalBath'] = (all_data['FullBath'] + 
                         0.5 * all_data['HalfBath'] +
                         all_data['BsmtFullBath'] + 
                         0.5 * all_data['BsmtHalfBath'])
print("✓ TotalBath créé (total salles de bain)")

# Âge de la maison
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
print("✓ HouseAge et RemodAge créés")

# Maison rénovée ou non
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
print("✓ IsRemodeled créé (0/1)")

# Score de qualité combiné
all_data['QualityScore'] = all_data['OverallQual'] * all_data['OverallCond']
print("✓ QualityScore créé")

# A un garage
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
print("✓ HasGarage créé (0/1)")

# A un sous-sol
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
print("✓ HasBsmt créé (0/1)")

# A une piscine
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
print("✓ HasPool créé (0/1)")

print(f"\nNombre total de features : {all_data.shape[1]}")

# ============================================
# 4. GESTION DES VALEURS ABERRANTES
# ============================================
print("\n[4] GESTION DES VALEURS ABERRANTES")
print("-" * 70)

# Suppression des maisons avec surface habitable anormalement grande
# (identifiées lors de l'exploration)
outliers_indices = all_data[(all_data['GrLivArea'] > 4000) & 
                             (all_data.index < len(train_df))].index

if len(outliers_indices) > 0:
    print(f"✓ {len(outliers_indices)} valeurs aberrantes détectées dans GrLivArea")
    all_data = all_data.drop(outliers_indices)
    y_train = y_train.drop(outliers_indices)
    print(f"✓ Après suppression : {all_data.shape}")

# ============================================
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================
print("\n[5] ENCODAGE DES VARIABLES CATÉGORIELLES")
print("-" * 70)

# Identification des colonnes catégorielles
categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()
print(f"Nombre de colonnes catégorielles : {len(categorical_cols)}")

# Encodage avec LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    all_data[col] = le.fit_transform(all_data[col].astype(str))
    label_encoders[col] = le

print(f"✓ {len(categorical_cols)} colonnes encodées")

# ============================================
# 6. TRANSFORMATION DE LA VARIABLE CIBLE
# ============================================
print("\n[6] TRANSFORMATION DE LA VARIABLE CIBLE")
print("-" * 70)

# Transformation log pour normaliser la distribution
y_train_log = np.log1p(y_train)

print(f"Prix avant transformation - Moyenne : ${y_train.mean():,.2f}")
print(f"Prix après log transform - Moyenne : {y_train_log.mean():.4f}")
print("✓ Transformation logarithmique appliquée (normalisation)")

# ============================================
# 7. SÉPARATION DES DONNÉES
# ============================================
print("\n[7] SÉPARATION DES DONNÉES")
print("-" * 70)

# Récupération train/test
train_size = len(y_train)
X_train_full = all_data[:train_size].copy()
X_test = all_data[train_size:].copy()

# Suppression de la colonne Id
if 'Id' in X_train_full.columns:
    X_train_full = X_train_full.drop(['Id'], axis=1)
if 'Id' in X_test.columns:
    X_test = X_test.drop(['Id'], axis=1)

print(f"✓ X_train_full : {X_train_full.shape}")
print(f"✓ X_test       : {X_test.shape}")
print(f"✓ y_train      : {y_train.shape}")
print(f"✓ y_train_log  : {y_train_log.shape}")

# Division train/validation (80/20)
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_full, y_train_log, test_size=0.2, random_state=42
)

print(f"\n✓ Train set      : {X_train.shape}")
print(f"✓ Validation set : {X_val.shape}")

# ============================================
# 8. SAUVEGARDE DES DONNÉES PRÉTRAITÉES
# ============================================
print("\n[8] SAUVEGARDE DES DONNÉES")
print("-" * 70)

# Création du dossier processed s'il n'existe pas
import os
if not os.path.exists('dataset/processed'):
    os.makedirs('dataset/processed')

# Sauvegarde
X_train.to_csv('dataset/processed/X_train.csv', index=False)
X_val.to_csv('dataset/processed/X_val.csv', index=False)
X_train_full.to_csv('dataset/processed/X_train_full.csv', index=False)
X_test.to_csv('dataset/processed/X_test.csv', index=False)

y_train_split.to_csv('dataset/processed/y_train.csv', index=False)
y_val.to_csv('dataset/processed/y_val.csv', index=False)
y_train_log.to_csv('dataset/processed/y_train_full.csv', index=False)

test_ids.to_csv('dataset/processed/test_ids.csv', index=False)

print("✓ X_train.csv")
print("✓ X_val.csv")
print("✓ X_train_full.csv")
print("✓ X_test.csv")
print("✓ y_train.csv")
print("✓ y_val.csv")
print("✓ y_train_full.csv")
print("✓ test_ids.csv")

# Sauvegarde des encoders
import pickle
with open('dataset/processed/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ label_encoders.pkl")

print("\n" + "=" * 70)
print(" " * 15 + "✓ PRÉTRAITEMENT TERMINÉ !")
print("=" * 70)
print("\nFichiers sauvegardés dans : dataset/processed/")
print("\nProchaine étape : Entraînement des modèles (03_train_models.py)")