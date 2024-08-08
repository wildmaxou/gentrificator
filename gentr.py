import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso



# Données sur le niveaux de vie par arondissement en 2020
data = {
    'Arrondissement': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Code_insee' :[75101, 75102, 75103, 75104, 75105, 75106, 75107, 75108, 75109, 75110, 75111, 75112, 75113, 75114, 75115, 75116, 75117, 75118, 75119, 75120],
    'Niveaux de vie / UC': [3130, 2978, 3056, 2878, 3178, 3743, 4227, 4085, 3329, 2476, 2583, 2575, 2130, 2520, 2899, 3793, 2948, 2034, 1795, 1892]
}
# Transformation en DF
niveaux_vie = pd.DataFrame(data)


# Machine learning
from sklearn.linear_model import Lasso
import numpy as np

@st.cache_data()
def load_and_transform_data():
  paris = pd.read_csv("https://raw.githubusercontent.com/Robinho67200/df_paris_immo/main/df_paris_prepared.csv")
  # Dummies
  dataframe = pd.get_dummies(data = paris, columns = ['code_postal', 'nature_mutation', 'mois_vente', 'type_local'])
  return dataframe



# Création des onglets
with st.sidebar:
  # Créez une barre de navigation latérale (sidebar) pour la sélection de la page
  page = st.radio("Sélectionnez une page", ["Le Gentrificateur", "Estimation prix immobilier"])

# Condition pour afficher le contenu de la page 1
if page == "Le Gentrificateur":
    #st.image("gentrificateur.png")

    # Choix de l'arrondissement
    arrondissement = st.selectbox(
        'Dans quel arrondisement voulez-vous déménager ?',
        (niveaux_vie["Arrondissement"]))

    # Nombre d'adultes dans le foyer
    # nb_adultes = st.number_input("Combien d'adultes êtes-vous ? ", min_value=1, max_value=4, step=1)


    # Initialisez les variables de session
    if 'nb_adultes' not in st.session_state:
        st.session_state.nb_adultes = 0
    if 'nb_enfant_moins_14' not in st.session_state:
        st.session_state.nb_enfant_moins_14 = 0
    if 'nb_enfant_plus_14' not in st.session_state:
        st.session_state.nb_enfant_plus_14 = 0

    # Initialisez les variables de session pour suivre les clics des boutons
    if 'clicked_adult' not in st.session_state:
        st.session_state.clicked_adult = False
    if 'clicked_baby' not in st.session_state:
        st.session_state.clicked_baby = False
    if 'clicked_girl' not in st.session_state:
        st.session_state.clicked_girl = False

    st.write("Combien d'adultes êtes-vous ? ")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(":red_haired_man:", key='adult1', disabled=st.session_state.clicked_adult):
            st.session_state.nb_adultes = 1
            st.session_state.clicked_adult = True
    with col2:
        if st.button(":red_haired_man::red_haired_man:", key='adult2', disabled=st.session_state.clicked_adult):
            st.session_state.nb_adultes = 2
            st.session_state.clicked_adult = True
    with col3:
        if st.button(":red_haired_man::red_haired_man::red_haired_man:", key='adult3', disabled=st.session_state.clicked_adult):
            st.session_state.nb_adultes = 3
            st.session_state.clicked_adult = True
    with col4:
        if st.button(":red_haired_man::red_haired_man::red_haired_man::red_haired_man:", key='adult4', disabled=st.session_state.clicked_adult):
            st.session_state.nb_adultes = 4
            st.session_state.clicked_adult = True

    st.write("Combien d'enfant de moins de 14 ans avez-vous ? ")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(":baby:", key='baby1', disabled=st.session_state.clicked_baby):
            st.session_state.nb_enfant_moins_14 = 1
            st.session_state.clicked_baby = True
    with col2:
        if st.button(":baby::baby:", key='baby2', disabled=st.session_state.clicked_baby):
            st.session_state.nb_enfant_moins_14 = 2
            st.session_state.clicked_baby = True
    with col3:
        if st.button(":baby::baby::baby:", key='baby3', disabled=st.session_state.clicked_baby):
            st.session_state.nb_enfant_moins_14 = 3
            st.session_state.clicked_baby = True
    with col4:
        if st.button(":baby::baby::baby::baby:", key='baby4', disabled=st.session_state.clicked_baby):
            st.session_state.nb_enfant_moins_14 = 4
            st.session_state.clicked_baby = True

    st.write("Combien d'enfant de plus de 14 ans avez-vous ? ")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(":girl:", key='girl1', disabled=st.session_state.clicked_girl):
            st.session_state.nb_enfant_plus_14 = 1
            st.session_state.clicked_girl = True
    with col2:
        if st.button(":girl::girl:", key='girl2', disabled=st.session_state.clicked_girl):
            st.session_state.nb_enfant_plus_14 = 2
            st.session_state.clicked_girl = True
    with col3:
        if st.button(":girl::girl::girl:", key='girl3', disabled=st.session_state.clicked_girl):
            st.session_state.nb_enfant_plus_14 = 3
            st.session_state.clicked_girl = True
    with col4:
        if st.button(":girl::girl::girl::girl:", key='girl4', disabled=st.session_state.clicked_girl):
            st.session_state.nb_enfant_plus_14 = 4
            st.session_state.clicked_girl = True

    st.write(f"Vous êtes {st.session_state.nb_adultes} adulte(s), {st.session_state.nb_enfant_moins_14} enfant(s) de moins de 14 ans et {st.session_state.nb_enfant_plus_14} enfant(s) de plus de 14 ans")


    #if st.button("⬇️ Prochaine étape ⬇️"):
      # Salaire par mois sur 12, 13, 14 mois
    if st.session_state.nb_adultes == 1 :
      salaires_1 = st.number_input("Quel est le salaire du 1er adulte ? ", min_value=0, step=1)
    elif st.session_state.nb_adultes == 2 :
      salaires_1 = st.number_input("Quel est le salaire du 1er adulte ? ", min_value=0, step=1)
      salaires_2 = st.number_input("Quel est le salaire du 2ème adulte ? ", min_value=0, step=1)
    elif st.session_state.nb_adultes == 3 :
      salaires_1 = st.number_input("Quel est le salaire du 1er adulte ? ", min_value=0, step=1)
      salaires_2 = st.number_input("Quel est le salaire du 2ème adulte ? ", min_value=0, step=1)
      salaires_3 = st.number_input("Quel est le salaire du 3ème adulte ? ", min_value=0, step=1)
    elif st.session_state.nb_adultes == 4 :
      salaires_1 = st.number_input("Quel est le salaire du 1er adulte ? ", min_value=0, step=1)
      salaires_2 = st.number_input("Quel est le salaire du 2ème adulte ? ", min_value=0, step=1)
      salaires_3 = st.number_input("Quel est le salaire du 3ème adulte ? ", min_value=0, step=1)
      salaires_4 = st.number_input("Quel est le salaire du 4ème adulte ? ", min_value=0, step=1)




    # Bouton pour afficher le résultat
    if st.button("Es-tu un Gentrificateur ? "):
      # Résultat du niveaux de vie
      if st.session_state.nb_adultes == 1 :
        resultat = (salaires_1) / ((st.session_state.nb_adultes) + (st.session_state.nb_enfant_plus_14*0.5) + (st.session_state.nb_enfant_moins_14*0.3))
      elif st.session_state.nb_adultes == 2 :
        resultat = (salaires_1 + salaires_2) / ((st.session_state.nb_adultes) + (st.session_state.nb_enfant_plus_14*0.5) + (st.session_state.nb_enfant_moins_14*0.3))
      elif st.session_state.nb_adultes == 3 :
        resultat = (salaires_1 + salaires_2 + salaires_3) / ((st.session_state.nb_adultes) + (st.session_state.nb_enfant_plus_14*0.5) + (st.session_state.nb_enfant_moins_14*0.3))
      elif st.session_state.nb_adultes == 4 :
        resultat = (salaires_1 + salaires_2 + salaires_3 + salaires_4) / ((st.session_state.nb_adultes) + (st.session_state.nb_enfant_plus_14*0.5) + (st.session_state.nb_enfant_moins_14*0.3))

      # Récupération du niveaux de vie de l'arrondissement
      niveaux_vie_arrondissement = niveaux_vie.loc[niveaux_vie["Arrondissement"] == arrondissement]["Niveaux de vie / UC"].values[0]

      # Comparaison
      if resultat > niveaux_vie_arrondissement :
        st.write("🧔‍♂️ Vous êtes un gentrificateur 👩")
      else :
        st.write(f"Eh non, il faut gagner plus de money 💵")

# Condition pour afficher le contenu de la page 2
elif page == "Estimation prix immobilier":
    st.header("🏠 Estimation prix immobilier 🏠")



    # Valeur par défaut des dummies

    code_postal_75001 = 0
    code_postal_75002 = 0
    code_postal_75003 = 0
    code_postal_75004 = 0
    code_postal_75005 = 0
    code_postal_75006 = 0
    code_postal_75007 = 0
    code_postal_75008 = 0
    code_postal_75009 = 0
    code_postal_75010 = 0
    code_postal_75011 = 0
    code_postal_75012 = 0
    code_postal_75013 = 0
    code_postal_75014 = 0
    code_postal_75015 = 0
    code_postal_75016 = 0
    code_postal_75017 = 0
    code_postal_75018 = 0
    code_postal_75019 = 0
    code_postal_75020 = 0
    nature_mutation_Ancien  = 0
    nature_mutation_Neuf = 0
    mois_vente_1 = 0
    mois_vente_2 = 0
    mois_vente_3 = 0
    mois_vente_4 = 0
    mois_vente_5 = 0
    mois_vente_6 = 0
    mois_vente_7 = 0
    mois_vente_8 = 0
    mois_vente_9 = 0
    mois_vente_10 = 0
    mois_vente_11 = 0
    mois_vente_12 = 0
    type_local_Appartement = 0
    type_local_Maison = 0


    # Sélection des valeurs des variables
    arrondissement = st.selectbox(
        'Dans quel arrondissement souhaitez-vous déménager ?',
        (niveaux_vie["Arrondissement"]))


    ancien_neuf = st.selectbox(
          'Neuf ou Ancien ?',
          ("Neuf", "Ancien"))


    appart_maison = st.selectbox(
          'Appartement ou Maison ?',
          ("Appartement", "Maison"))

    
    mois = st.selectbox(
          "L'achat se fera en :",
          ("Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"))  
    surface_reelle_bati = st.number_input("Surface réelle :", min_value=1, step=1)
    nombre_pieces_principales = st.number_input("Nombre de pieces:", min_value=1, step=1)
    surface_moyenne_piece = st.number_input("Surface moyenne d'une piece :", min_value=1, step=1)
    

    # Bouton pour afficher le résultat
    if st.button("Envie d'une estimation ?"):
      
      # Appelez la fonction pour charger et transformer les données
      dataframe = load_and_transform_data()
      # Splitting data into features and target
      X = dataframe[['surface_reelle_bati',
          'nombre_pieces_principales',
          'surface_moyenne_piece', 'code_postal_75001', 'code_postal_75002',
          'code_postal_75003', 'code_postal_75004', 'code_postal_75005',
          'code_postal_75006', 'code_postal_75007', 'code_postal_75008',
          'code_postal_75009', 'code_postal_75010', 'code_postal_75011',
          'code_postal_75012', 'code_postal_75013', 'code_postal_75014',
          'code_postal_75015', 'code_postal_75016', 'code_postal_75017',
          'code_postal_75018', 'code_postal_75019', 'code_postal_75020',
          'nature_mutation_Ancien', 'nature_mutation_Neuf', 'mois_vente_1',
          'mois_vente_2', 'mois_vente_3', 'mois_vente_4', 'mois_vente_5',
          'mois_vente_6', 'mois_vente_7', 'mois_vente_8', 'mois_vente_9',
          'mois_vente_10', 'mois_vente_11', 'mois_vente_12',
          'type_local_Appartement', 'type_local_Maison']]

      y = dataframe['prix_2023']


      # Création du modèle
      model = Lasso(3.4304692863149193)
      # Entrainement du modèle
      model.fit(X, y)










      if mois == "Janvier" :
        mois_vente_1 = 1
      elif mois == "Fevrier" :
        mois_vente_2 = 1
      elif mois == "Mars" :
        mois_vente_3 = 1
      elif mois == "Avril" :
        mois_vente_4 = 1
      elif mois == "Mai" :
        mois_vente_5 = 1
      elif mois == "Juin" :
        mois_vente_6 = 1
      elif mois == "Juillet" :
        mois_vente_7 = 1
      elif mois == "Août" :
        mois_vente_8 = 1
      elif mois == "Septembre" :
        mois_vente_9 = 1
      elif mois == "Octobre" :
        mois_vente_10 = 1
      elif mois == "Novembre" :
        mois_vente_11 = 1
      elif mois == "Décembre" :
        mois_vente_12 = 1

      if appart_maison == "Appartement" :
        type_local_Appartement = 1
      elif appart_maison == "Maison":
        type_local_Maison = 1

      if ancien_neuf == "Neuf" :
        nature_mutation_Neuf = 1
      elif ancien_neuf == "Ancien":
        nature_mutation_Ancien = 1



      if arrondissement == 1 :
        code_postal_75001 = 1
      elif arrondissement == 2 :
        code_postal_75002 = 1
      elif arrondissement == 3 :
        code_postal_75003 = 1
      elif arrondissement == 4 :
        code_postal_75004 = 1
      elif arrondissement == 5 :
        code_postal_75005 = 1
      elif arrondissement == 6 :
        code_postal_75006 = 1
      elif arrondissement == 7 :
        code_postal_75007 = 1
      elif arrondissement == 8 :
        code_postal_75008 = 1
      elif arrondissement == 9 :
        code_postal_75009 = 1
      elif arrondissement == 10 :
        code_postal_75010 = 1
      elif arrondissement == 11 :
        code_postal_75011 = 1
      elif arrondissement == 12 :
        code_postal_75012 = 1
      elif arrondissement == 13 :
        code_postal_75013 = 1
      elif arrondissement == 14 :
        code_postal_75014 = 1
      elif arrondissement == 15 :
        code_postal_75015 = 1
      elif arrondissement == 16 :
        code_postal_75016 = 1
      elif arrondissement == 17 :
        code_postal_75017 = 1
      elif arrondissement == 18 :
        code_postal_75018 = 1
      elif arrondissement == 19 :
        code_postal_75019 = 1
      elif arrondissement == 20 :
        code_postal_75020 = 1

      my_data = np.array([surface_reelle_bati,
      nombre_pieces_principales,
      surface_moyenne_piece, code_postal_75001, code_postal_75002,
      code_postal_75003, code_postal_75004, code_postal_75005,
      code_postal_75006, code_postal_75007, code_postal_75008,
      code_postal_75009, code_postal_75010, code_postal_75011,
      code_postal_75012, code_postal_75013, code_postal_75014,
      code_postal_75015, code_postal_75016, code_postal_75017,
      code_postal_75018, code_postal_75019, code_postal_75020,
      nature_mutation_Ancien, nature_mutation_Neuf, mois_vente_1,
      mois_vente_2, mois_vente_3, mois_vente_4, mois_vente_5,
      mois_vente_6, mois_vente_7, mois_vente_8, mois_vente_9,
      mois_vente_10, mois_vente_11, mois_vente_12,
      type_local_Appartement, type_local_Maison]).reshape(1,39)
      prediction = round(model.predict(my_data)[0])
      st.write(f"Ce logement vous coûtera {prediction} euros")

