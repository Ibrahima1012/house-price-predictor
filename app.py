# app.py
# Application web pour la prédiction du prix de l'immobilier

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Immobilier",
    page_icon="🏠",
    layout="wide"
)

# ============================================
# CHARGEMENT DES MODÈLES (CACHE)
# ============================================
@st.cache_resource
def load_models():
    """Charge tous les modèles entraînés"""
    with open('models/xgboost.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/ridge_regression.pkl', 'rb') as f:
        ridge_model = pickle.load(f)
    with open('dataset/processed/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return xgb_model, rf_model, ridge_model, encoders

@st.cache_data
def load_data():
    """Charge les données d'exemple"""
    X_train = pd.read_csv('dataset/processed/X_train.csv')
    return X_train

# Chargement
try:
    xgb_model, rf_model, ridge_model, encoders = load_models()
    X_train = load_data()
    feature_names = X_train.columns.tolist()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles : {e}")
    st.stop()

# ============================================
# TITRE ET DESCRIPTION
# ============================================
st.title("🏠 Prédiction du Prix de l'Immobilier")
st.markdown("""
Cette application utilise **Machine Learning** pour prédire le prix d'une maison 
en fonction de ses caractéristiques.
""")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", 
                        ["🔮 Prédiction Simple", 
                         "⚙️ Prédiction Avancée", 
                         "📊 Analyse du Modèle"])

# ============================================
# PAGE 1 : PRÉDICTION SIMPLE
# ============================================
if page == "🔮 Prédiction Simple":
    st.header("🔮 Prédiction Simple")
    st.markdown("Entrez les caractéristiques principales de la maison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Caractéristiques Générales")
        overall_qual = st.slider("Qualité générale (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Surface habitable (sq ft)", 500, 5000, 1500)
        year_built = st.number_input("Année de construction", 1870, 2023, 2000)
        
    with col2:
        st.subheader("Sous-sol et Garage")
        total_bsmt_sf = st.number_input("Surface sous-sol (sq ft)", 0, 3000, 1000)
        garage_cars = st.slider("Places de garage", 0, 4, 2)
        garage_area = st.number_input("Surface garage (sq ft)", 0, 1500, 500)
        
    with col3:
        st.subheader("Autres")
        full_bath = st.slider("Nombre de salles de bain", 0, 4, 2)
        bedroom_abv_gr = st.slider("Nombre de chambres", 0, 8, 3)
        kitchen_qual = st.select_slider("Qualité cuisine", 
                                        options=['Poor', 'Fair', 'Typical', 'Good', 'Excellent'],
                                        value='Typical')
    
    if st.button("🔮 Prédire le Prix", type="primary"):
        # Créer un vecteur de features avec des valeurs par défaut
        input_data = pd.DataFrame([X_train.iloc[0]])  # Copier la structure
        
        # Mettre à jour les valeurs importantes
        input_data['OverallQual'] = overall_qual
        input_data['GrLivArea'] = gr_liv_area
        input_data['YearBuilt'] = year_built
        input_data['TotalBsmtSF'] = total_bsmt_sf
        input_data['GarageCars'] = garage_cars
        input_data['GarageArea'] = garage_area
        input_data['FullBath'] = full_bath
        input_data['BedroomAbvGr'] = bedroom_abv_gr
        
        # Features dérivées
        input_data['TotalSF'] = total_bsmt_sf + gr_liv_area
        input_data['TotalBath'] = full_bath + 0.5
        input_data['HouseAge'] = 2023 - year_built
        
        # Prédiction
        pred_xgb = np.expm1(xgb_model.predict(input_data))[0]
        pred_rf = np.expm1(rf_model.predict(input_data))[0]
        pred_ridge = np.expm1(ridge_model.predict(input_data))[0]
        pred_ensemble = (0.5 * pred_xgb + 0.3 * pred_rf + 0.2 * pred_ridge)
        
        # Affichage des résultats
        st.success("✅ Prédiction réussie !")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Prix Estimé (Ensemble)", f"${pred_ensemble:,.0f}")
        col2.metric("XGBoost", f"${pred_xgb:,.0f}")
        col3.metric("Random Forest", f"${pred_rf:,.0f}")
        col4.metric("Ridge", f"${pred_ridge:,.0f}")
        
        # Graphique de comparaison
        fig = go.Figure(data=[
            go.Bar(name='Modèles', x=['XGBoost', 'Random Forest', 'Ridge', 'Ensemble'],
                   y=[pred_xgb, pred_rf, pred_ridge, pred_ensemble],
                   marker_color=['purple', 'coral', 'green', 'gold'])
        ])
        fig.update_layout(title="Comparaison des prédictions",
                         yaxis_title="Prix ($)",
                         showlegend=False,
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Intervalle de confiance approximatif
        std_dev = np.std([pred_xgb, pred_rf, pred_ridge])
        lower_bound = pred_ensemble - 1.96 * std_dev
        upper_bound = pred_ensemble + 1.96 * std_dev
        
        st.info(f"📊 Intervalle de confiance (95%) : ${lower_bound:,.0f} - ${upper_bound:,.0f}")

# ============================================
# PAGE 2 : PRÉDICTION AVANCÉE
# ============================================
elif page == "⚙️ Prédiction Avancée":
    st.header("⚙️ Prédiction Avancée")
    st.markdown("Entrez toutes les caractéristiques pour une prédiction plus précise")
    
    # Upload CSV
    st.subheader("📁 Option 1 : Télécharger un fichier CSV")
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV avec les features", type=['csv'])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier chargé : {input_df.shape[0]} maisons")
            
            # Vérifier que les colonnes correspondent
            missing_cols = set(feature_names) - set(input_df.columns)
            if missing_cols:
                st.warning(f"⚠️ Colonnes manquantes : {missing_cols}")
                st.info("Les colonnes manquantes seront remplies avec des valeurs par défaut")
                for col in missing_cols:
                    input_df[col] = X_train[col].median()
            
            # Réorganiser les colonnes
            input_df = input_df[feature_names]
            
            if st.button("🔮 Prédire pour toutes les maisons"):
                predictions = np.expm1(xgb_model.predict(input_df))
                
                result_df = input_df.copy()
                result_df['Prix_Prédit'] = predictions
                
                st.success(f"✅ {len(predictions)} prédictions générées !")
                
                # Statistiques
                col1, col2, col3 = st.columns(3)
                col1.metric("Prix Moyen", f"${predictions.mean():,.0f}")
                col2.metric("Prix Min", f"${predictions.min():,.0f}")
                col3.metric("Prix Max", f"${predictions.max():,.0f}")
                
                # Afficher les résultats
                st.dataframe(result_df[['Prix_Prédit']].head(20), use_container_width=True)
                
                # Télécharger les résultats
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Graphique de distribution
                fig = px.histogram(predictions, nbins=50,
                                  title="Distribution des prix prédits",
                                  labels={'value': 'Prix ($)', 'count': 'Fréquence'})
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    
    # Formulaire manuel complet
    st.subheader("✍️ Option 2 : Saisie manuelle complète")
    st.info("Cette fonctionnalité nécessite la saisie de toutes les features. Utilisez plutôt la prédiction simple.")

# ============================================
# PAGE 3 : ANALYSE DU MODÈLE
# ============================================
elif page == "📊 Analyse du Modèle":
    st.header("📊 Analyse et Performance du Modèle")
    
    # Charger les résultats de comparaison
    try:
        comparison = pd.read_csv('models/model_comparison.csv')
        
        st.subheader("🏆 Comparaison des Modèles")
        
        # Tableau
        st.dataframe(comparison, use_container_width=True)
        
        # Graphiques de comparaison
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(comparison, x='model', y='r2',
                         title="R² Score par Modèle",
                         labels={'r2': 'R² Score', 'model': 'Modèle'},
                         color='r2',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(comparison, x='model', y='rmse',
                         title="RMSE par Modèle (plus bas = meilleur)",
                         labels={'rmse': 'RMSE', 'model': 'Modèle'},
                         color='rmse',
                         color_continuous_scale='Reds')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Feature Importance
        st.subheader("🎯 Importance des Features (XGBoost)")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig3 = px.bar(feature_importance, x='importance', y='feature',
                     orientation='h',
                     title="Top 20 des Features les Plus Importantes",
                     labels={'importance': 'Importance', 'feature': 'Feature'},
                     color='importance',
                     color_continuous_scale='Blues')
        fig3.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Insights
        st.subheader("💡 Insights")
        st.markdown(f"""
        - **Meilleur modèle** : {comparison.iloc[0]['model']} avec un R² de {comparison.iloc[0]['r2']:.4f}
        - **Précision** : Le modèle explique {comparison.iloc[0]['r2']*100:.2f}% de la variance des prix
        - **Erreur moyenne** : ${comparison.iloc[0]['mae']:,.2f}
        - **Feature la plus importante** : {feature_importance.iloc[0]['feature']}
        """)
        
    except Exception as e:
        st.error(f"❌ Impossible de charger les résultats : {e}")

# ============================================
# FOOTER
# ============================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 À propos")
st.sidebar.info("""
**Projet de Prédiction Immobilière**

Modèles utilisés :
- XGBoost
- Random Forest
- Ridge Regression

Créé avec Streamlit et scikit-learn
""")