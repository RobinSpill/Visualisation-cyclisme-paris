# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:59:56 2020

@author: robin
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import contextily
import geopandas as gpd
from shapely.geometry import Point

st.beta_set_page_config(layout="wide")
st.title('Visualisation cyclisme Paris')

from PIL import Image
image = Image.open('compteur.jpg')
st.image(image, caption="Photo d'un compteur de passage", use_column_width=False)
st.header('Jeu de données brut')
df3 = pd.read_csv('comptage-velo-donnees-compteurs.csv', sep = ';')
st.write(df3.head())
df2 = pd.read_csv("Vélibrécent.csv")
st.header("Jeu de données transformé")
st.write(df2.head())
# Visualisation géographique
df = pd.read_csv("VélibStreamlit3.csv")

def map(data, lon, lat, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "longitude" : lon,
            "latitude" : lat,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["Longitude", "Latitude"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))
    
st.header("Implantation des compteurs et affluence moyenne par heure")
heure_choisie = st.slider("Heure d'affluence:", 0, 23)
    
df['HH']= df['HH'].astype('int')
data = df[df["HH"] == heure_choisie]
midpoint = (np.average(data["Latitude"]), np.average(data["Longitude"]))

st.write("**Affluence cycliste entre %i:00 et %i:00**" % (heure_choisie, (heure_choisie + 1) % 24))
map(data, midpoint[1], midpoint[0], 11)


#Deuxième visualisation
df1 = pd.read_csv('Vélib.csv')
Moyennehoraire = df1.groupby(['Nom du compteur','Longitude','Latitude'], as_index = False).agg({'Comptage horaire' : 'mean'}).sort_values(by = 'Comptage horaire', ascending = False)

col1, col2 = st.beta_columns(2)
fig1 = sns.catplot(x= 'Nom du compteur', y = 'Comptage horaire', data = Moyennehoraire.head(10), kind = 'bar', height = 6, aspect = 2)
fig1.set_xticklabels(rotation = 45)
fig2 = sns.catplot(x= 'Nom du compteur', y = 'Comptage horaire', data = Moyennehoraire.tail(10), kind = 'bar', height = 6, aspect = 2)
fig2.set_xticklabels(rotation = 45)
with col1:
    st.header("10 compteurs avec le plus d'affluence")
    st.pyplot(fig1)
    crs={'init':'epsg:4326'}
    
    gMoyennehorairegps = gpd.GeoDataFrame(Moyennehoraire.head(11), geometry=gpd.points_from_xy(Moyennehoraire.head(11).Longitude, Moyennehoraire.head(11).Latitude))
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(111)
    gMoyennehorairegps.plot( ax=ax, legend = True, legend_kwds={'label': "Moyenne d'affluence"}, markersize = 500)
    
    contextily.add_basemap(ax, crs = crs)
    st.header('Sur la carte')
    st.write(fig)
    
with col2:
    st.header("10 compteurs avec le moins d'affluence")
    st.pyplot(fig2)
    crs={'init':'epsg:4326'}
    
    gMoyennehorairegps = gpd.GeoDataFrame(Moyennehoraire.head(11), geometry=gpd.points_from_xy(Moyennehoraire.tail(11).Longitude, Moyennehoraire.tail(11).Latitude))
    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(111)
    gMoyennehorairegps.plot(ax=ax, legend = True, legend_kwds={'label': "Moyenne d'affluence"}, markersize = 500, facecolor = None)
    contextily.add_basemap(ax, crs = crs)
    st.write(fig)

#Deuxieme visualisation bis

Moyenneparjour = df1.groupby(['Jour de la semaine'], as_index = False).agg({"Comptage horaire" : 'mean'})
g= sns.catplot(x= 'Jour de la semaine', y = 'Comptage horaire', data = Moyenneparjour, kind = 'bar', aspect = 4)
plt.xticks(np.arange(7),['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']);
st.header('Affluence moyenne des compteurs en fonction du jour de la semaine')
st.pyplot(g)
                                               
#Troisième visualisation
st.header("Visualisation moyenne par heure au cours d'une journée")
import plotly_express as px
df2 = pd.read_csv("Vélibrécent.csv")

col1, col2, col3 = st.beta_columns(3)

Moyennejour = df2.groupby(['Heure', 'Week-end'], as_index = False).agg({'Comptage horaire' : 'mean'})
fig1 = px.bar(Moyennejour, x='Heure', y='Comptage horaire', color = 'Week-end')
Moyenneweekend = Moyennejour[Moyennejour['Week-end']== 1]
fig2 = px.bar(Moyenneweekend, x='Heure', y= 'Comptage horaire')
Moyennesemaine = Moyennejour[Moyennejour['Week-end']==0]
fig3 = px.bar(Moyennesemaine, x='Heure', y='Comptage horaire')
with col3 :
    st.header('Tous les jours')
    barplot_chart = st.write(fig1)
with col2 : 
    st.header('Le Week-end')
    barplot_chart= st.write(fig2)
with col1 :
    st.header('La semaine')
    barplot_chart = st.write(fig3)

#Quatrième visualisation

st.header('Visualisation sur un compteur')
compteur = df1['Nom du compteur'].unique()
compteur_choisi = st.multiselect('Choisir un compteur', compteur)


jour = df1['Jour'].unique()
jour_choisi = st.multiselect('Choisir un jour', jour)

mois = df1['Mois'].unique()
mois_choisi = st.multiselect('Choisir un mois', mois)

Uncompteur = df1[(df1['Nom du compteur'].isin(compteur_choisi)) & (df1['Jour'].isin(jour_choisi)) & (df1['Mois'].isin(mois_choisi))]


data = pd.DataFrame({'Heure': Uncompteur['Heure'], 'Affluence': Uncompteur['Comptage horaire']})

data = data.rename(columns={'Heure':'index'}).set_index('index')

row1_1, row1_2 = st.beta_columns((2,3))

with row1_1:
    st.header("Affluence au cours de la journée")
    st.area_chart(data)

with row1_2:
    st.header("Le voici sur la carte")
    midpoint1 = (np.average(Uncompteur["Latitude"]), np.average(Uncompteur["Longitude"]))
    carte = pd.DataFrame({'lat': midpoint1[0], 'lon' : midpoint1[1]}, index = [0,1])
    st.map(carte)
    

#Cinquième visualisation

Moyenneparmois = df2[df2['Annee']==2020].groupby(['Mois'], as_index = False).agg({"Comptage horaire" : 'mean'})
fig = px.bar(Moyenneparmois, x='Mois', y= 'Comptage horaire',)
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = np.arange(1, 13),
        ticktext = ['Janvier', 'Février', 'Mars','Avril','Mai','Juin','Juillet','Aout','Septembre','Octobre','Novembre','Décembre']), 
    autosize=False,
    width=1900,
    height=500,)

st.header("Affluence moyenne au cours de l'année 2020")
barplot_chart = st.write(fig)

#Sixième visualisation
dfconf1 = pd.read_csv('Vélibconf1.csv')
Moyennejourconf1 = dfconf1.groupby(['Date','J_x'], as_index = False).agg({'Comptage horaire' : 'mean'})
dfconf2 = pd.read_csv('Vélibconf2.csv')
Moyennejourconf2 = dfconf2.groupby(['Date','J'], as_index = False).agg({'Comptage horaire' : 'mean'})

col1, col2 = st.beta_columns(2)
fig1 = sns.catplot(x= 'Date', y = 'Comptage horaire', data = Moyennejourconf1, kind = 'bar',aspect = 4, sharey = True)
fig1.set_xticklabels(rotation = 90)
fig2 = sns.catplot(x= 'Date', y = 'Comptage horaire', data = Moyennejourconf2, kind = 'bar',aspect = 4, sharey = True)
fig2.set_xticklabels(rotation = 90)
st.header("Affluence moyenne par jour des compteurs lors du premier confinement")
st.pyplot(fig1)
st.header("Affluence moyenne par jour des compteurs lors du deuxième confinement")
st.pyplot(fig2)

#Septieme visualisation
st.header("Comparaison dans le temps de l'affluence moyenne des compteurs entre le premier et le deuxième confinement")

data = pd.DataFrame({'J': Moyennejourconf2['J'], 'Conf1': Moyennejourconf1['Comptage horaire'], 'Conf2': Moyennejourconf2['Comptage horaire']})

data = data.rename(columns={'J':'index'}).set_index('index')

st.area_chart(data)

#Huitième visualisation

Moyennejourconf3 = dfconf1.groupby(['J_x', 'Nom du compteur'], as_index = False).agg({'Comptage horaire' : 'mean'})

Moyennejourconf4 = dfconf2.groupby(['J', 'Nom du compteur'], as_index = False).agg({'Comptage horaire' : 'mean'})

compteur = dfconf2['Nom du compteur'].unique()
#compteur_choisi = st.multiselect('Choisir un compteur', compteur)
Uncompteur1 = Moyennejourconf3[(Moyennejourconf3['Nom du compteur'].isin(compteur_choisi))]
Uncompteur2 = Moyennejourconf4[(Moyennejourconf4['Nom du compteur'].isin(compteur_choisi))]

data1 = pd.DataFrame({'J': Uncompteur2['J'], 'Conf1': Uncompteur1['Comptage horaire'], 'Conf2' : Uncompteur2['Comptage horaire']})

data1 = data1.rename(columns={'J':'index'}).set_index('index')



