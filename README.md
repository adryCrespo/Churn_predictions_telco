# Bajas de cliente en un telco
![](https://github.com/adryCrespo/Churn_predictions_telco/blob/main/st_folder/tfno2.jpeg)
Este es un proyecto de data science que predice la baja de clientes en una compañia telco. Para ello cuenta con datos sobre el cliente, como pueden ser el numero de minutos que habla por el dia o el número de llamadas a atencion al cliente que realiza.

Para hacer estas predicciones se ha desarrollado un modelo de clasificacion de machine learning. Este modelo nos da la probabilidad que el cliente se de baja de la compañia.

Además, se ha creado un applicacion dashboard en streamlit para usar este modelo de forma interactiva. Puedes consultar la aplicacion en el siguiente [link](https://adrycrespo-churn-predictions-telco-app-churn-i0rbfx.streamlit.app/)

-------------
## datos y exploracion
Los datos originales proceden de un concurso de kaggle ( [link]( https://www.kaggle.com/competitions/customer-churn-prediction-2020/data) ). El
concurso evaluaba la precision de los modelos en la salida de clientes usando la métrica accuracy. Los datos procedian exclusivamente de usuarios de USA.

La mayor parte de informacion se basa en métricas sobre el consumo telefónico anual del cliente. Por ejemplo, el número de minutos que habla al año en una franja horaria.
También habia otras métricas como el estado de procedencia del cliente o el numero de llamadas a atencion al cliente que se han realizado.

El dataset es relativamente pequeño (miles de datos), y como he dicho arriba, la mayoria de la variables son númericas, que se distribuyen con una distribución normal. El mayor problema que presenta el dataset es el desbalanceo de la clase target: sólo el 14% de los clientes del dataset se han dado de baja.

He realizado una exploracion de datos, para mejorar mi conocimiento del dataset y crear nuevas variables. Las 3 conclusiones mas interesantes:
- He visto el numero de personas que se dan de baja en la compañia es muy alto en el decil superior **minutos hablados por la mañana** (de 250 minutos a 350 minutos).
-  Si el **numero de servicio de llamadas** es igual a 4 o superior se dispara la probabilidad de dejar la compañia
-  El **estado** en el que vive el cliente influye mucho en la probabilidad de baja en la compañia

Para ver estos resultados con más detenimiento ver el  notebook "Target EDA.ipynb"

## Modelo 
He visto probado diferentes algoritmos pero el mejor es que ha funcionado es XGBOOST con un grid search. 

**Transformación de variables**
Las variables númericas se han estandarizado y en las variables categoricas las he tratado con One-hot-encoding.
Además, he realizado winsorizacion a algunas de las variables númericas para tratar atipicos.

**Modelo**:
Como he comentado anteriormente, he elegido el algoritmo con un grid search usando auc-roc como métrica evaluadora. Para tratar el desambalaceo de los datos he usado una estragia de penalizacion por peso en contra parte de realizar la estrategia estandar de remuestreo.
Por usabilidad, he implementado un pipeline con todas las transformaciones y el modelo y la he cargado en un archivo pickle.

Para ver los detalles, ver el notebook "codigo productivo.ipynb"

## App streamlit
Se ha creado una webapp con el paque te streamlit. Esta aplicacion usa el pipeline con el modelo y sus transformaciones para realizar las predicciones.
La [aplicacion](https://adrycrespo-churn-predictions-telco-app-churn-i0rbfx.streamlit.app/) se ha publicado en el servidor de streamlit.

En esta aplicación, el usuario mete sus datos de manera manual con pestañas y sliders. En cambio, el estado del usario se selecciona con un mapa interactivo. 
Este mapa se ha realizado con el paquete folium.

El codigo de la app esta en el archivo "app_churn.py"
