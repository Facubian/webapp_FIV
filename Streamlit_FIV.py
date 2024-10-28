import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import distribution

data= pd.read_csv("data_etiquetada.csv")
data = data[data['amh'] <= 15]
data = data[data['total rfa'] <= 30]

st.markdown(
    """
    # Embryoxite LAB
    Esta web es un experimento para mostrar distintos modelos de inteligencia artificial (IA) para predecir el número de ovocitos posibles a recuperar en pacientes que realizan tratamiento de reproducción asistida.    
    **Modelos actuales:**

    - Ensamble de regresores logísticos v1
    - Super Vector Classifier v1

    ### ¿Queres saber mas de nosotros?

    - Visita la pagina oficial de [Embryoxite](https://embryoxite.life/)
    - Revisa nuestro [Linkedin](https://www.linkedin.com/company/embryoxite)
"""
)

st.write(f'<h2 class="big-font"> Simulación de Recuperacion de ovocitos </h2>', unsafe_allow_html=True)
st.write(f'<p class="big-font"> Podemos observar la distribucion de los datos con los que fueron entrenados nuestros modelos</p>', unsafe_allow_html=True)
st.write(f'<p class="big-font"> Seleccione los valores de cada variable para observar el posible resultado que obtendria si el paciente tiene esas características</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])


with col1:
    
    variable="edad paciente"
    st.write(f'<h4 class="big-font"> Ingrese la edad  del paciente: </h4>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    age = st.slider("",18, 50, label_visibility="collapsed") 
    distribution(data,variable,age) #
    st.write(f"Usted ingreso que el paciente tiene: {age} años")

with col2:
    variable= "amh"
    st.write(f'<h4 class="big-font">Ingrese un valor de Antimulleriana en sangre: </h4>', unsafe_allow_html=True)
    amh_log = st.slider(f'<h2 class="big-font">Ingrese cantidades obtenidas de \n la hormona antimuleriana en sangre: </h2>', 0.01, 13.0, label_visibility="collapsed")
    distribution(data,variable,amh_log) 
    st.write(f"Usted ingreso un valor de Antimulleriana en sangre de: {amh_log} ng/ml")

with col3:
    variable= "total rfa"
    st.write(f'<h4 class="big-font">Ingrese el número foliculos antrales del paciente:  </h4>', unsafe_allow_html=True)
    total_rfa = col3.slider(f"Ingrese el total de recuentos de foliculos antrales", 1, 25, label_visibility="collapsed")
    distribution(data,variable,total_rfa) 
    st.write(f"Usted ingreso una cantidad de foliculos antrales: {total_rfa}")


df_pred = pd.DataFrame([[age,amh_log,total_rfa]])#

u_edad = 34.60
u_amh = 2.52929
u_rfa = 9.9528
s_edad = 5.4431
s_amh = 2.38858
s_rfa = 5.32975

amhn = (amh_log-u_amh)/s_amh
edadn = (age-u_edad)/s_edad
rfan = (total_rfa-u_rfa)/s_rfa

df_pred2= pd.DataFrame([[edadn, amhn, rfan]])#

columns= ['edad paciente','amh_log','total rfa']
df_pred.columns = columns

columns= ['edad paciente','amh',' total rfa']
df_pred2.columns = columns
#Transformaciones de los datos
#Transformación Logarítmica
def transform(data):
    return np.log(data)

#A los primeros modelos se le aplica un transformacion logarítmica al dato 
df_pred['amh_log'] = df_pred['amh_log'].apply(transform)

#Carga de modelos
model1 = joblib.load('RL1_model.pkl')
model2 = joblib.load('RL2_model.pkl')

model3 = joblib.load('RF_model.pkl')


# Ejemplo de función para obtener predicciones de los modelos
def obtener_predicciones(df_pred):
    prediction1 = model1.predict(df_pred)
    prediction2 = model2.predict(df_pred)
    
    if prediction1 == 0:
        result = 0
    elif prediction1 == 1:
        if prediction2 == 1:
            result = 2
        elif prediction2 == 0:
            result = 1
    else:
        result = None  # Manejar casos inesperados
    
    return prediction1, prediction2, result

# Interfaz de la aplicación con Streamlit
def main():
    st.write(f'<h2 class="big-font"> Predicciones de los modelos </h2>', unsafe_allow_html=True)
    
    # Aquí deberías tener el código para cargar y preparar tus datos de entrada (df_pred)
    # df_pred = cargar_datos_para_prediccion()  # Ejemplo
    
    #if st.button('Predict'):
    prediction1, prediction2, result = obtener_predicciones(df_pred)
    #Ensamble de modelos 
    
    prediction3= model3.predict(df_pred2)
    # Convertir el resultado a texto para mostrar en la interfaz
    if result == 0:
        result_str = "Menor o igual a 4"
    elif result == 1:
        result_str = "5-9"
    elif result == 2:
        result_str = "Mayor a 9"
    else:
        result_str = "No disponible"
     # Convertir el resultado del tercer modelo a texto para mostrar en la interfaz
    
    if prediction3 == 1:
        result_str2 = "Menor o igual a 4"
    elif prediction3 == 2:
        result_str2 = "5-9"
    elif prediction3 == 3:
        result_str2 = "Mayor a 9"
    else:
        result_str2 = "No disponible" 
    
    # Mostrar los resultados en la interfaz
    st.markdown(f'<p class="big-font"> El proyecto utiliza dos modelos de regresion lineal para realizar la prediccion, acorde al resultado de los modelos se toma una decision final de a que grupo pertenecen</p>', unsafe_allow_html=True)
    # Mostrar los datos seleccionados en la interfaz
    st.markdown("### Datos seleccionados: ")
    st.markdown(f"- **Edad:** {age} años")
    st.markdown(f"- **Amh:** {amh_log} ng/ml")
    st.markdown(f"- **N° Folículos antrales:** {total_rfa}")    
    
    col1, col2,col3 = st.columns(3)

    with col1:
        st.markdown("### Ensamble de Regresores Logísticos (LR)")
        st.markdown(f'Predicción para el primer modelo,  paciente con mayor probabilidad de ovocitos capturados en el rango: **<span style="color:red;">{"Menor o igual a 4" if prediction1 == 0 else "Mayor a 4"}</span>**', unsafe_allow_html=True)
        st.markdown(f'Predicción para el segundo modelo, paciente con mayor probabilidad de ovocitos capturados en el rango: **<span style="color:red;">{"Igual o menor a 9" if prediction2 == 0 else "Mayor a 9"}</span>**', unsafe_allow_html=True)
        st.markdown(f' __Prediccion final__ para el ensamble de modelos(LR) el paciente tiene mayores probabilidades de caer en el rango de ovocitos capturados {"entre" if result == 1 else ""}: **<span style="color:red;">{result_str}</span>**', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Random Forest (RF)")
        st.markdown(f'Para el modelo RF el paciente tiene mayores probabilidades de caer en el rango de ovocitos capturados {"entre" if prediction3=="2" else ""}: **<span style="color:red;">{result_str2}</span>**', unsafe_allow_html=True)

    with col3:
        st.markdown("### Super Vector Classifier (SVC)")
        st.markdown(f'Para el modelo SCV el paciente tiene mayores probabilidades de caer en el rango de ovocitos capturados {"entre" if prediction3=="2" else ""}: **<span style="color:red;">{result_str2}</span>**', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
