# 01_exploration.py
# Script d'exploration des données immobilières

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 70)
print(" " * 15 + "EXPLORATION DES DONNÉES IMMOBILIÈRES")
print("=" * 70)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n[1] CHARGEMENT DES DONNÉES")
print("-" * 70)

try:
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    print(f"✓ Données d'entraînement : {train_df.shape[0]} lignes, {train_df.shape[1]} colonnes")
    print(f"✓ Données de test : {test_df.shape[0]} lignes, {test_df.shape[1]} colonnes")
except FileNotFoundError:
    print("❌ ERREUR : Fichiers non trouvés !")
    print("   Assurez-vous que train.csv et test.csv sont dans le dossier 'dataset/'")
    exit()

# ============================================
# 2. APERÇU DES DONNÉES
# ============================================
print("\n[2] APERÇU DES PREMIÈRES LIGNES")
print("-" * 70)
print(train_df.head())

print("\n[3] INFORMATIONS SUR LES COLONNES")
print("-" * 70)
print(f"Nombre total de colonnes : {len(train_df.columns)}")
print(f"Colonnes numériques : {len(train_df.select_dtypes(include=[np.number]).columns)}")
print(f"Colonnes catégorielles : {len(train_df.select_dtypes(include=['object']).columns)}")

# ============================================
# 3. ANALYSE DE LA VARIABLE CIBLE
# ============================================
print("\n[4] ANALYSE DE LA VARIABLE CIBLE : SalePrice")
print("-" * 70)
print(f"Prix moyen        : ${train_df['SalePrice'].mean():,.2f}")
print(f"Prix médian       : ${train_df['SalePrice'].median():,.2f}")
print(f"Prix minimum      : ${train_df['SalePrice'].min():,.2f}")
print(f"Prix maximum      : ${train_df['SalePrice'].max():,.2f}")
print(f"Écart-type        : ${train_df['SalePrice'].std():,.2f}")

# Visualisation de la distribution des prix
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analyse de la Distribution des Prix', fontsize=16, fontweight='bold')

# Histogramme
axes[0, 0].hist(train_df['SalePrice'], bins=50, edgecolor='black', color='skyblue')
axes[0, 0].set_xlabel('Prix de vente ($)', fontsize=12)
axes[0, 0].set_ylabel('Fréquence', fontsize=12)
axes[0, 0].set_title('Distribution des prix')
axes[0, 0].axvline(train_df['SalePrice'].mean(), color='red', linestyle='--', 
                    label=f"Moyenne: ${train_df['SalePrice'].mean():,.0f}")
axes[0, 0].legend()

# Distribution log (plus normale)
axes[0, 1].hist(np.log(train_df['SalePrice']), bins=50, edgecolor='black', color='lightgreen')
axes[0, 1].set_xlabel('Log(Prix de vente)', fontsize=12)
axes[0, 1].set_ylabel('Fréquence', fontsize=12)
axes[0, 1].set_title('Distribution log des prix (transformation)')

# Box plot
axes[1, 0].boxplot(train_df['SalePrice'], vert=True)
axes[1, 0].set_ylabel('Prix de vente ($)', fontsize=12)
axes[1, 0].set_title('Box Plot - Détection des valeurs aberrantes')

# QQ Plot pour normalité
from scipy import stats
stats.probplot(train_df['SalePrice'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot - Test de normalité')

plt.tight_layout()
plt.show()

# ============================================
# 4. VALEURS MANQUANTES
# ============================================
print("\n[5] ANALYSE DES VALEURS MANQUANTES")
print("-" * 70)

missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)

if len(missing_train) > 0:
    missing_percent = (missing_train / len(train_df)) * 100
    
    missing_df = pd.DataFrame({
        'Nombre manquant': missing_train.values,
        'Pourcentage (%)': missing_percent.values
    }, index=missing_train.index)
    
    print(missing_df.head(20))
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    top_missing = missing_df.head(20)
    plt.barh(range(len(top_missing)), top_missing['Pourcentage (%)'], color='coral')
    plt.yticks(range(len(top_missing)), top_missing.index)
    plt.xlabel('Pourcentage de valeurs manquantes (%)', fontsize=12)
    plt.title('Top 20 des colonnes avec valeurs manquantes', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("✓ Aucune valeur manquante détectée !")

# ============================================
# 5. CORRÉLATIONS AVEC LE PRIX
# ============================================
print("\n[6] CORRÉLATIONS AVEC LE PRIX DE VENTE")
print("-" * 70)

# Calcul des corrélations
numeric_features = train_df.select_dtypes(include=[np.number])
correlations = numeric_features.corr()['SalePrice'].sort_values(ascending=False)

print("Top 10 des corrélations positives :")
print(correlations.head(11)[1:])  # Exclure SalePrice lui-même

print("\nTop 10 des corrélations négatives :")
print(correlations.tail(10))

# Heatmap des corrélations
plt.figure(figsize=(14, 10))
top_corr_features = correlations.head(16).index
corr_matrix = train_df[top_corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de corrélation - Top 15 features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 6. ANALYSE DES FEATURES IMPORTANTES
# ============================================
print("\n[7] ANALYSE DES FEATURES CLÉS")
print("-" * 70)

# Relation entre surface habitable et prix
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Relations entre features clés et prix', fontsize=16, fontweight='bold')

# Surface habitable vs Prix
axes[0, 0].scatter(train_df['GrLivArea'], train_df['SalePrice'], alpha=0.5)
axes[0, 0].set_xlabel('Surface habitable (sq ft)')
axes[0, 0].set_ylabel('Prix de vente ($)')
axes[0, 0].set_title('Surface habitable vs Prix')

# Qualité générale vs Prix
quality_price = train_df.groupby('OverallQual')['SalePrice'].mean()
axes[0, 1].bar(quality_price.index, quality_price.values, color='steelblue')
axes[0, 1].set_xlabel('Qualité générale (1-10)')
axes[0, 1].set_ylabel('Prix moyen ($)')
axes[0, 1].set_title('Qualité vs Prix moyen')

# Année de construction vs Prix
year_price = train_df.groupby('YearBuilt')['SalePrice'].mean()
axes[1, 0].plot(year_price.index, year_price.values, marker='o', markersize=3)
axes[1, 0].set_xlabel('Année de construction')
axes[1, 0].set_ylabel('Prix moyen ($)')
axes[1, 0].set_title('Année de construction vs Prix')

# Nombre de chambres vs Prix
bedroom_price = train_df.groupby('BedroomAbvGr')['SalePrice'].mean()
axes[1, 1].bar(bedroom_price.index, bedroom_price.values, color='lightcoral')
axes[1, 1].set_xlabel('Nombre de chambres')
axes[1, 1].set_ylabel('Prix moyen ($)')
axes[1, 1].set_title('Chambres vs Prix moyen')

plt.tight_layout()
plt.show()

# ============================================
# 7. STATISTIQUES DESCRIPTIVES
# ============================================
print("\n[8] STATISTIQUES DESCRIPTIVES (VARIABLES NUMÉRIQUES)")
print("-" * 70)
print(train_df.describe().T)

print("\n" + "=" * 70)
print(" " * 20 + "✓ EXPLORATION TERMINÉE !")
print("=" * 70)
print("\nProchaine étape : Prétraitement des données (02_preprocessing.py)")