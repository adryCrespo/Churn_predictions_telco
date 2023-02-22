#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:
#streamlit run app_churn.py

import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import cloudpickle
from streamlit_echarts import st_echarts

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

#CONFIGURACION DE LA PÁGINA
st.set_page_config(
     page_title = 'prediccion baja cliente',
     layout = 'wide')
     


#@st.cache()
def get_df_estados():
    
    #diccionario
    dict_real_state = 'diccionario_state.pickle'
    ruta_pipe_state = './data/' + dict_real_state
    with open(ruta_pipe_state, mode='rb') as file:
       medias_dict = cloudpickle.load(file)  
    
    abrev = list(medias_dict.keys() )
    valores = list(medias_dict.values() )
    df_dict = pd.DataFrame({'Abbreviation':abrev,'valores':valores})
    
    #diccionario
    dict_real_state = 'lista_estados.pickle'
    ruta_pipe_state = './data/' + dict_real_state
    with open(ruta_pipe_state, mode='rb') as file:
       lista_estados = cloudpickle.load(file) 
    
    df_estados = lista_estados.reset_index().merge(df_dict, how='left', on='Abbreviation')
    return df_estados.rename(columns={'State Name':'state_name'})
    
#@st.cache()
def get_pipeline():
    ruta_pipe_ejecucion = './data/' +'pipe_ejecucion.pickle'
    with open(ruta_pipe_ejecucion, mode='rb') as file:
        pipe_ejecucion = cloudpickle.load(file)
    return pipe_ejecucion
 
# INICIAR VARIABLES 
df_estados = get_df_estados()
lista = df_estados['state_name'].unique().tolist()
df_index = df_estados.set_index('state_name')
state_name = 'Utah'
pipe_ejecucion = get_pipeline()


                      
                        
#SIDEBAR
with st.sidebar:
    international_plan = st.selectbox('international_plan:',['yes','no'])
    voice_mail_plan = st.selectbox('voice_mail_plan:',['yes','no'])
    number_customer_service_calls = st.slider('numero de llamadas a servicio al cliente: ', 0, 4,0)
    total_day_minutes = st.slider('total_day_minutes ', 0, 400,200)
    total_eve_minutes = st.slider('total_eve_minutes ', 0, 400,40)
    total_night_minutes = st.slider('total_night_minutes ', 0, 400,20)
    total_intl_minutes = st.slider('total_intl_minutes ', 0, 20,0)
    total_intl_calls = 10
    ratio_state = df_index.loc[state_name,'valores'] if state_name in lista else 0.14

# DATASET
#Crear el registro
registro = pd.DataFrame({'international_plan':international_plan,
                         'number_customer_service_calls':number_customer_service_calls,
                         'total_day_minutes': total_day_minutes,
                         'total_eve_minutes': total_eve_minutes,
                         'total_intl_calls': total_intl_calls,
                         'total_intl_minutes': total_intl_minutes,
                         'total_night_minutes': total_night_minutes,
                         'voice_mail_plan': voice_mail_plan,
                         'ratio_state': ratio_state}
                        ,index=[0])


#MAPA
st.title('Prediccion de baja de clientes')

st.write("Selecciona un estado en el mapa")
mapUSA = folium.Map(location=[38, -96.5], zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')

custom_scale = (df_estados['valores'].quantile((0,0.2,0.4,0.6,0.8,1))).tolist()

choropleth =folium.Choropleth(
            geo_data=r'./data/us-state-boundaries.geojson',
            data=df_estados,
            columns=['state_name', 'valores'],  #Here we tell folium to get the county fips and plot new_cases_7days metric for each county
            key_on='feature.properties.name', #Here we grab the geometries/county boundaries from the geojson file using the key 'coty_code' which is the same as county fips
            threshold_scale=custom_scale ,
            legend_name='New Cases Past 7 Days (Per 100K Population) '
            )
choropleth.geojson.add_to(mapUSA)
for feature in choropleth.geojson.data['features']:
        state_name = feature['properties']['name']
        feature['properties']['valor'] = 'Valor: ' + '{:,}'.format( round(df_index.loc[state_name, ['valores']][0],4) ) if state_name in lista else ' '

choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name'], labels=False)
    )



#CONFIGURACION MODELO
new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 
'early_stopping_rounds', 'max_cat_to_onehot', 'max_leaves', 'sampling_method',
 'feature_types','max_cat_threshold']

for attr in new_attrs:
    setattr(pipe_ejecucion[1], attr, None)

# MAIN

st_map = st_folium(mapUSA, width=700, height=450)
state_name = ''
if st_map['last_active_drawing']:
        state_name = st_map['last_active_drawing']['properties']['name']
        y =  df_index.loc[state_name, ['valores']][0]
        registro['ratio_state'] = y

# prediccion
scoring = pipe_ejecucion.predict_proba(registro)[:, 1]
x = round(min(scoring[0]*100,100),2)


ead_options = {
            "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
            "series": [
                {
                    "name": "probabilidad salida de cliente",
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 1,
                        },
                    },
                    "progress": {"show": "true", "width": 1},
                    "detail": {"valueAnimation": "true", "formatter": "{value}"},
                    "data": [{"value": x, "name": "salida cliente"}],
                }
            ],
        }


# datos
st.write("Estado seleccionado: "+ state_name if state_name in lista else "Estado no seleccionado")
st.write("Probabilidad de salida del cliente: " +str(round(x,2))+"%"  )
st_echarts(options=ead_options, width="110%", key=1)

st.dataframe(registro)