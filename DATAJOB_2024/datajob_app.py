import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px




### FONCTION DE CHARGEMENT DES DATASETS
# permet de ne pas rammer pour entrainement des diffs algo 
# les datasets ne changent pas en fonction des modèles

@st.cache_data
def load_data():
    data=pd.read_csv('dataset_pred.csv')  
    return data 
df=load_data()

# MISE EN PAGE DE L'APPLI
st.image("datajob_logo_rouge.PNG")
st.title( "METIERS DE LA DATASCIENCE" )   
st. sidebar.title( " Sommaire" )                                         

pages=["Exploration des données", "DataVisualisation", "Modélisation prédictive"  ]
page=st.sidebar.radio("", pages )

# PAGE 1 PRESENTATION DATASETS
if page == pages[ 0 ] : 
  st.write("## _Présentation des données_" )
  st.dataframe(df.head( 10 ) )
  st.write("#### Taille du dataset")
  st.write( df.shape )  
  st.write("#### Description des variables catégorielles")
  if st.checkbox("Liste des valeurs manquantes") :
     st.dataframe(df.isna().sum()) 
  if st.checkbox("Liste des métiers") :
     st.dataframe(df["Current_role"].unique()) 
  if st.checkbox("Liste des compétences") :
     st.dataframe(df.columns[6:])
  if st.checkbox("Liste des niveaux d'études") :
     st.dataframe(df["EducationLevel"].unique())
  if st.checkbox("Répartition des tranches d'âge") :
     st.dataframe(df["Age"].value_counts())


# PAGE 2 PRESENTATION VISUELLE DES DATA
if page == pages[ 1 ] : 
    st.write( "## _Visualisation des données_" )
    import plotly.express as px
    import plotly.graph_objects  as  go


# distribution des postes
    valeurs=df['Current_role'].value_counts()
    class_names=['Data Scientist','Data Analyst',
                'Software Engineer','Research Scientist']
    fig=go.Figure( data= [ go.Pie( labels=class_names , 
                                    values=valeurs , 
                                    hole=0.5 ) ] )
    fig.update_layout( xaxis_title=None , yaxis_title=None ,
                                legend_font_size=17,font_size=17 ,
                                title_text='Répartition moyenne des sondés' ,
                                        title_x=0.1  ,
                                        title_font_color="#e12929",
                                        title_font_size=25,
                                        height=400)
    st.plotly_chart( fig )         

# répartition des niveaux de formation des sondés
    fig1=px.bar( df, 
             y="EducationLevel", 
             x="Current_role",
             color="EducationLevel",
             color_discrete_map={'Other':'#f0bd37',
                         'Master':'#ffacd0',
                         'Bachelor':'#6fa8dc',
                         'Professional':'#cbdff7',
                         'Doctoral':'#dcedc1',
                         'NotAnswer':'#f60303'},
             height=600,
             width=700)
    fig1.update_layout( xaxis_title=None )
    fig1.update_layout( yaxis_title=None )
    fig1.update_layout( legend=None , legend_font_size=17 )
    fig1.update_layout( font_size=17 )   
    fig1.update_yaxes( tickfont_size=17 )
    fig1.update_layout( title_text='Répartition des niveaux de formation des sondés', 
                                    title_x=0.1,
                                    title_font_color="#e12929",
                                    title_font_size=25,
                                    height=500 )

    fig1.update_yaxes( tickfont_size=17 )
    fig1.update_xaxes( tickfont_size=18 )
    st.plotly_chart( fig1 )

# répartition des ages des sondés

    fig2=px.bar( df, 
             barmode='group', 
             x='Current_role',
             color='Age',height=600,
             width=700,color_discrete_map={'45-59':'#5b9cea',
                         '30-44':'#9ac2f3',
                         '60+':'#217ce7',
                         '18-29':'#cbdff7'} )
    fig2.update_layout(xaxis_title=None)
    fig2.update_layout(yaxis_title=None)
    fig2.update_layout(legend_font_size=17)
    fig2.update_layout(font_size=17)   
    fig2.update_yaxes(tickfont_size=1)
    fig2.update_layout(title_text='Répartition moyenne des âges par postes occupés', 
                                    title_x=0.1,
                                    title_font_color="#e12929", 
                                    title_font_size=25,
                                    height=500)   
    fig2.update_xaxes(tickfont_size=18)
    fig2.update_yaxes(tickfont_size=10)
    st.plotly_chart(fig2)

    



#PAGE 3 MODELISATION ET AUTOMATISATION SUR DATASET PREDICTIF
if page == pages[ 2 ] : 
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier 
  from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix,accuracy_score
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import LabelEncoder,MinMaxScaler
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC


### MODELISATION

# train_test_split
  target=df['Current_role']
  feats=df.drop(['Current_role'],axis=1)
  X_train,X_test,y_train,y_test=train_test_split(feats, target, test_size=0.2)

#  sépa vars nums et catés
  num_train=X_train.select_dtypes(include='int64')
  num_test=X_test.select_dtypes(include='int64')
  cat_train=X_train.select_dtypes(include='object')
  cat_test=X_test.select_dtypes(include='object')

# encodage de target
  label=LabelEncoder()
  y_train=label.fit_transform(y_train)
  y_test=label.transform(y_test)

# encodage feats nums
  scaler=StandardScaler()
  num_train=scaler.fit_transform(num_train)
  num_test=scaler.transform(num_test)

# encodage feats catés
  cat_train=pd.get_dummies(cat_train)
  cat_test=pd.get_dummies(cat_test)

# reconstitution des datasets encodé
  X_train=np.concatenate([num_train,cat_train],axis=1)
  X_test=np.concatenate([num_test,cat_test],axis=1)

# normalisation des données (pour ne pas avoir de valeurs négatives pour entrainement)
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

### MISE EN PLACE DES AFFICHAGES POUR LA MODELISATION
# à mettre dans l'ordre d'affichage souhaité

  st.write('### CORRESPONDANCE CLASSES / METIERS')
  correspondance={'Classe n° 0':'Data Scientist',
                        'Classe n° 1':'Data Analyst',
                        'Classe n°2':'Software Engineer',
                        'Classe n°3':'Research Scientist'}
  st.image("correspondance_classes.PNG")

  class_names=['Data Scientist','Data Analyst',
                'Software Engineer','Research Scientist']

## GESTION DES MODELES  
# Liste des modèles de prédiction
  model_choice=['DecisionTreeClassifier','RandomForestClassifier',
                        'LogisticRegression','KNeighborsClassifier',"Support Vector Machine (SVM)"]
# Titre de la selectbox de choix du modèle (sur la gauche de la page)
  st.sidebar.subheader('Choix du modèle')
# Mise en place de la sélection du modèle
  model_select=st.sidebar.selectbox("Modèle choisi", model_choice)

## GESTION DES METRIQUES  
# Liste des métriques
  metric_choice=['Accuracy', 'Matrice de confusion','Rapport de classification']
# Titre de la selectbox de choix des métriques (sur la gauche de la page)
  st.sidebar.subheader('Choix du métrique de performance')
# Mise en place de la sélection du métrique
  metric_select= st.sidebar.selectbox('Métrique choisi',metric_choice)


### MISE EN PLACE D'UNE F° DE PREDICTION
  def prediction(classifier):
    
# RandomForestClassifier
    if classifier == 'RandomForestClassifier':
       st.sidebar.subheader("Hyperparamètres")
       n_estimators = st.sidebar.number_input("Nombre d'arbres", 
                                           min_value=100, 
                                           max_value=5000, 
                                           step = 10)
       max_depth = st.sidebar.number_input("Profondeur des arbres", 
                                        min_value=4, 
                                        max_value=20, 
                                        step = 1)
       model=RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth)
        

# KNeighborsClassifier   
    elif classifier == 'KNeighborsClassifier':
        st.sidebar.subheader("Hyperparamètres")
        n_neighbors=st.sidebar.number_input("Profondeur des arbres", 
                                        min_value=4, 
                                        max_value=20)
        metric=st.sidebar.radio("Type de mesure de la distance des voisins", ( 'euclidean',  'manhattan',  'minkowski'))
        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     metric=metric)

        
# Support Vector Machine (SVM)  
    elif classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Hyperparamètres")
        C = st.sidebar.number_input("Paramètre de régularisation",
                                        min_value=0.01, 
                                        max_value=10.0)
        kernel = st.sidebar.radio("Type de noyau", 
                                      ("linear", "poly", "rbf", "sigmoid"))
        gamma = st.sidebar.radio("Coefficient du noyau", 
                                     ("scale", "auto"))

        model = SVC(C =C ,
                    kernel=kernel,
                    gamma =gamma)

# LogisticRegression      
    elif classifier == 'LogisticRegression':
        st.sidebar.subheader("Hyperparamètres")
        C= st.sidebar.number_input("Paramètre de régularisation", 
                                        min_value=0.01, 
                                        max_value=10.0, 
                                        step = 0.3)
        max_iter = st.sidebar.slider("Nombre maximum d'itération", 
                                           min_value=100, 
                                           max_value=500)
        model = LogisticRegression(C=C,
                                   max_iter=max_iter)
        
# DecisionTreeClassifier       
    elif classifier == 'DecisionTreeClassifier':
        st.sidebar.subheader("Hyperparamètres")
        max_depth = st.sidebar.number_input("Profondeur des arbres", 
                                        min_value=3, 
                                        max_value=20, 
                                        step = 1)
        min_samples_split=st.sidebar.slider(
            "Nombre minimum d'échantillons requis pour diviser un nœud interne", 
                                        min_value=2, 
                                        max_value=8)
        criterion=st.sidebar.radio("Fonction de mesure de la qualité d'une division", 
                                     ("gini", "log_loss","entropy"),
                                     key="criterion")
        model = DecisionTreeClassifier(max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       criterion=criterion)
    model.fit(X_train, y_train)
    return model

# IMPORTATION DES MODELES ENTRAINES ISSUS DU NOTEBOOK
  import joblib
  def load_model(model_choice):
        if model_choice == 'RandomForestClassifier':
            model=joblib.load('RandomForestClassifier')
        elif model_choice == 'LogisticRegression':
            model=joblib.load('LogisticRegression')
        elif model_choice == 'KNeighborsClassifier':
            model= joblib.load('KNeighborsClassifier')
        elif model_choice == 'DecisionTreeClassifier':
            model=joblib.load('DecisionTreeClassifier')
        elif model_choice == 'SVC':
            model= joblib.load('SVC')
        return model


# Affection du modèle entrainé aux différents modèles et prédiction
  model=prediction(model_select)
  y_pred=model.predict(X_test)


# Calcul et affichage des métriques choisis
  if metric_select=='Accuracy':
      accuracy=model.score(X_test,y_test)
      st.write("")
      st.write("")
      st.write("")
      st.write("##### __*ACCURACY DU MODELE CHOISI*__ ")
      st.write("**ACCURACY** = Mesure de l'exactitude globale des prédictions d'un modèle en calculant le rapport entre les échantillons correctement classés et le nombre total d'échantillons.")
      st.subheader(accuracy)
  elif metric_select=='Matrice de confusion':
      st.write("")
      st.write("")
      st.write("##### __*COMMENT LIRE UNE MATRICE DE CONFUSION*__ ")
      st.write("La matrice de confusion est un outil de mesure de la performance des modèles de classification. Elle résume ici de manière graphique les valeurs absolues des données prédictives et réelles")
      st.image("matrice_1.png.~tmp")
      st.write("") 
      st.write("") 
      st.write("") 
      conf_matrix = confusion_matrix(y_test, y_pred)
      fig, ax = plt.subplots(figsize=(10,8))
      st.write("##### __*MATRICE DE CONFUSION DU MODELE CHOISI*__ ")
      sns.heatmap(conf_matrix, 
                  annot=True, 
                  fmt="d", 
                  cmap="Blues", 
                  xticklabels=class_names,
                  annot_kws = {'fontsize': 13},
                  yticklabels=class_names)
      plt.xlabel('Prédictions',color="red",fontsize=19)
      plt.ylabel('Réalité',color="red",fontsize=19)
      plt.yticks(rotation=35,fontsize=15)
      plt.xticks(rotation=35,fontsize=15)
      st.write(fig)
  elif metric_select=='Rapport de classification':
      st.write("")
      st.write("")
      st.write("")
      st.write("##### __*COMMENT LIRE UN RAPPORT DE CLASSIFICATION*__ ")
      st.write("Le rapport de classification est un outil utilisé dans l'apprentissage automatique pour évaluer les performances d'un modèle de classification . Il affiche diverses mesures telles que la précision, le rappel et le score F1 pour chaque classe du problème de classification.")
      st.image("rapport_classification.PNG")
      st.write("**RECALL** = Correspond au taux de vrais positifs, et mesure le rapport entre les vrais positifs et le nombre total d'échantillons positifs réels.")   
      st.write("**PRECISION** = Cette mesure quantifie le rapport entre les vrais positifs et le nombre total de prédictions positives.")
      st.write("**F1 SCORE** = Mesure qui combine précision et rappel en une seule valeur. Elle mesure la précision globale du classificateur. Le score le plus élevé est 1 et le score le plus bas est 0.")
      st.write("**SUPPORT** = Nombre d'occurences réelles de la classe dans l'ensemble de donnée spécifié")
      st.write("")
      st.write("")
      st.write("")
      st.write("##### __*RAPPORT DE CLASSIFICATION DU MODELE CHOISI*__ ")
      report=pd.DataFrame(classification_report(y_test,
                                          y_pred=model.predict(X_test),
                                          target_names=class_names,
                                          output_dict=True)).iloc[:5,:4].transpose()
      st.table(data=report)





















 




  