import pandas as pd
import streamlit as st
import joblib
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

modelo = joblib.load("modelo1.pkl")
# ----------------------------------------------------------------

# Titulo
st.title('Predicción de Adaptabilidad de Cultivo según condiciones de Clima y Suelo')
st.markdown('***')
# ----------------------------------------------------------------

# Recupere todos los datos de la tabla en un DataFrame de pandas
data = pd.read_csv('CSVs\data.csv')

# elaboracion de panel de consulta de requerimientos de cultivos
st.markdown("<h2 style='text-align: center; color: green;'>Consulta de Requerimientos de Cultivos</h2>",
            unsafe_allow_html=True)

cultivo = st.selectbox(
    'Selecciona una Cultivo',
    sorted(data['label'].unique(), reverse=False))

st.write(f"Has seleccionado: {cultivo}")
# ----------------------------------------------------------------

# requerimientos de N, P, K
st.write('### Para el Cultivo de ', cultivo,
         ' los Requerimientos de Nitrogeno, Fósforo y Potásio son:')

x = data[data['label'] == cultivo]
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>NITROGENO</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = x['N'].min()

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = x['N'].mean()

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = x['N'].max()

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>FÓSFORO</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = x['P'].min()

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = x['P'].mean()

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = x['P'].max()

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>POTÁSIO</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = x['K'].min()

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = x['K'].mean()

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = x['K'].max()

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

# requerimientos de Suelo y Clima
st.write('### Para el Cultivo de ', cultivo,
         ' los Requerimientos Climáticos y de Suelo son:')
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>TEMPERATURA</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = round(x['temperature'].min(), 2)

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = round(x['temperature'].mean(), 2)

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = round(x['temperature'].max(), 2)

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>HUMEDAD</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = round(x['humidity'].min(), 2)

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = round(x['humidity'].mean(), 2)

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = round(x['humidity'].max(), 2)

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>PH DEL SUELO</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = round(x['ph'].min(), 2)

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = round(x['ph'].mean(), 2)

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = round(x['ph'].max(), 2)

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

st.markdown("<p style='text-align: center;'>PRECIPITACIÓN</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    minimo = round(x['rainfall'].min(), 2)

    st.write('Mínimo: ', minimo)

# rasultado de promedio
with col2:
    medio = round(x['rainfall'].mean(), 2)

    st.write('Promedio: ', medio)

# rasultado de maximo
with col3:
    maximo = round(x['rainfall'].max(), 2)

    st.write('Máximo: ', maximo)
# ----------------------------------------------------------------

# elaboracion de un modelo de machine learning para predecir la adaptabilidad de un determinado cultivo a las condiciones propuestas
st.markdown("<h2 style='text-align: center; color: green;'>Predictor de un determinado cultivo a las condiciones propuestas</h2>",
            unsafe_allow_html=True)

st.write('A contonuación introduzca la disponibilidad de nutrientes (N, P, K) y las condiciones Climatológicas y de Suelo de su terreno.')

# ----------------------------------------------------------------

st.markdown("<p style='font-size: 25px; text-align: center;'>Disponibilidad de Nutrientes</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3 = st.columns(3)

# rasultado de minimo
with col1:
    st.write('#### Nitrógeno')
    nitrogeno = int(st.text_input(
        "(Entre " + str(data['N'].min()) + " y " + str(data['N'].max()) + ")", value="0"))

    st.write('Valor ingresado:', nitrogeno)

# rasultado de promedio
with col2:
    st.write('#### Fósforo')
    fosforo = int(st.text_input("(Entre " +
                  str(data['P'].min()) + " y " + str(data['P'].max()) + ")", value="0"))

    st.write('Valor ingresado:', fosforo)

# rasultado de maximo
with col3:
    st.write('#### Potásio')
    potasio = int(st.text_input("(Entre " +
                  str(data['K'].min()) + " y " + str(data['K'].max()) + ")", value="0"))

    st.write('Valor ingresado:', potasio)
# ----------------------------------------------------------------

st.markdown("<p style='font-size: 25px; text-align: center;'>Condiciones Climatológicas y de Suelo</p>",
            unsafe_allow_html=True)

# Divide la pantalla en tres columnas
col1, col2, col3, col4 = st.columns(4)

# rasultado de minimo
with col1:
    st.write('#### Temperatura')
    temperatura = float(st.text_input("(Entre " + str(round(
        data['temperature'].min(), 2)) + " y " + str(round(data['temperature'].max(), 2)) + ")", value="0"))

    st.write('Valor ingresado:', temperatura)

# rasultado de promedio
with col2:
    st.write('#### Humedad')
    humedad = float(st.text_input("(Entre " + str(round(
        data['humidity'].min(), 2)) + " y " + str(round(data['humidity'].max(), 2)) + ")", value="0"))

    st.write('Valor ingresado:', humedad)

# rasultado de maximo
with col3:
    st.write('#### pH')
    ph = float(st.text_input("(Entre " + str(round(
        data['ph'].min(), 2)) + " y " + str(round(data['ph'].max(), 2)) + ")", value="0"))

    st.write('Valor ingresado:', ph)

# rasultado de maximo
with col4:
    st.write('#### Precipitación')
    precipitacion = float(st.text_input("(Entre " + str(round(
        data['rainfall'].min(), 2)) + " y " + str(round(data['rainfall'].max(), 2)) + ")", value="0"))

    st.write('Valor ingresado:', precipitacion)
# ----------------------------------------------------------------

label = "prueba"  # por defecto

# tienes el siguiente DataFrame
x_prueba = pd.DataFrame(
    columns=('N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'))

# tienes los siguientes datos
datos = (nitrogeno, fosforo, potasio, temperatura,
         humedad, ph, precipitacion, label)

x_prueba.loc[0] = datos
y_pred = modelo.predict(x_prueba.drop(['label'], axis=1))

st.write(
    '## De acuerdo con la información suministradas el cultivo que mejor se adaptaría a dichas condiciones es: ', y_pred[0])
# ----------------------------------------------------------------

st.markdown('### A continuacion te invitamos a explorar otras alternativas de cultivos que pudieran adaptarse a las condiciones propuestas.')


def recommend_crops(data_file='data_features.csv', input_label=None, top_n=4):
    """
    Recomienda cultivos basándose en la similitud del coseno entre las características del cultivo.

    Parámetros:
    data_file (cadena): ruta al archivo CSV que contiene funciones de recorte.
    input_label (str): Etiqueta del cultivo para el que se necesitan recomendaciones.
    top_n (int): número de cultivos recomendados para devolver.

    Devoluciones:
    Marco de datos: Top N cultivos recomendados junto con sus puntuaciones de similitud.
    """
    # cargar la data
    data1 = pd.read_csv(data_file)

    # crear matriz de caracteristicas
    tfidv = TfidfVectorizer(min_df=2, max_df=0.7,
                            token_pattern=r'\b[a-zA-Z0-9]\w+\b')
    data_vector = tfidv.fit_transform(data1['features'])

    data_vector_df = pd.DataFrame(data_vector.toarray(),
                                  index=data1['label'], columns=tfidv.get_feature_names_out())

    # calcular la similitud de coseno de la matriz de caracteristicas de los cultivos
    vector_similitud_coseno = cosine_similarity(data_vector_df.values)

    cos_sim_df = pd.DataFrame(vector_similitud_coseno,
                              index=data_vector_df.index, columns=data_vector_df.index)

    # Obtener las similitudes para un cultiuvo específico
    if input_label is None:
        # la primera etiqueta única es la que nos interesa
        input_label = data1['label'].unique()[0]

    cultivo_simil = cos_sim_df.loc[input_label]

    # Ordena las similitudes en orden descendente.
    simil_ordenada = cultivo_simil.sort_values(ascending=False)

    # Obtenr el top N mas similares de los cultivos recomendados
    top_results = simil_ordenada.head(top_n).reset_index()

    res = top_results['label'][1:4]

    res_list = res.tolist()
    res_string = ', '.join(map(str, res_list))

    return res_string


resultado = recommend_crops('CSVs/data_features.csv', y_pred[0])
st.write(
    '### también se pueden adaptar los siguientes cultivos: ' + resultado + '.')
