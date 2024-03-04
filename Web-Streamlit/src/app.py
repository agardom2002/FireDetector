import av
import os
import re
import cv2
import nltk
import base64
import tempfile
import unidecode
import numpy as np
import streamlit as st

from gtts import gTTS
from ultralytics import YOLO
from nltk.corpus import stopwords
from MandarCorreo import enviar_correo
from nltk.tokenize import word_tokenize
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# -- CONFIGURACION DE LA PAGINA --

# Cambiamos el icono y nombre de la pestaña
st.set_page_config( 
    page_title="FireDetector", 
    page_icon="🔥",)

# Creamos una funcion para cargar el archivo CSS local (style.css)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Llamamos a la funcion (local_css) para cargar el archivo CSS
local_css("Web-Streamlit/src/style.css")

# Cambiamos los estilos de los botones
st.markdown(
    """
    <style>
    button.myButton {
        border-color: black;
        background-color: white;
        color: black;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -- INICIALIZACION DE VARIABLES --

# Cargamos el modelo de detección de fuego
model = YOLO("Web-Streamlit/model/best.pt")

# Diccionario para los idiomas
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
}

# Diccionario para la alarma según idioma
texto_audio = {"es": "ALERTA! FUEGO DETECTADO!", "en": "Alert, fire detected", "fr": "Alerte, incendie détecté",
               "de": "Alarm, Feuer erkannt", "it": "Allerta, rilevato incendio"}

no_detecta = {"es": "Lo siento, no te he entendido.", "en": "I'm sorry, I didn't understand you.", "fr": "Je suis désolé, je ne t'ai pas compris.",
               "de": "Es tut mir leid, ich habe dich nicht verstanden.", "it": "Mi dispiace, non ti ho capito."}

# Inicializacion del array que guardara los apartados procesados con NPL
pre_titles = []

# Array que contiene los nombres de los apartados de la pagina
tab_titles = ["Introducción", "Configuración", "Detección", "Alarma"]

# -- INICIALIZACION DE ESTADOS DE SESION -- 

# Guardamos en una variable el estado de la sesion
state = st.session_state

# Mediante un estado de sesion (descarga) 
# comprobamos si se han realizado las descargas de NLTK
if 'descarga' not in state:
    state.descarga = True
    nltk.download('punkt')
    nltk.download('stopwords')

# Inicializamos el estado de sesion para los idiomas
if 'langs' not in state:
    state.langs = "Spanish"

# Inicializamos el estado de sesion para el correo
if 'email' not in state:
    state.email = False

# -- INTRODUCCION --

# Funcion que muestra todo el contenido del apartado 'Introduccion'
def tab_introduccion():
    st.subheader("Introducción: ", divider = "red")    
    st.write("""
        🔥 ¡Atención a todos los amantes de la tecnología y la seguridad! 🔥 ¿Estás buscando una manera innovadora de proteger nuestros preciosos recursos naturales? ¡Entonces, este es el proyecto que estabas esperando!

        🚒 Presentamos nuestro increíble Trabajo de Fin de Máster en Inteligencia Artificial y Big Data: ¡La solución definitiva para la detección de incendios en áreas naturales!
            
        🌳 En un mundo donde los incendios forestales representan una seria amenaza para el medio ambiente y la seguridad pública, ¡nuestro proyecto brinda una solución inteligente y efectiva!
            
        💡 Con la combinación de algoritmos de Inteligencia Artificial y análisis de Big Data, nuestro sistema puede detectar la presencia de fuego en áreas naturales de manera precisa y oportuna. ¡No más preocupaciones por incendios no detectados!
            
        ✅ **¿Por qué elegir nuestra solución?**
            
        - **Eficiencia:** Nuestro sistema utiliza algoritmos avanzados para analizar imágenes y datos en tiempo real, lo que permite una detección temprana y rápida de incendios.
            
        - **Precisión:** Gracias a la inteligencia artificial, nuestro sistema reduce al mínimo los falsos positivos, garantizando una detección confiable.
            
        - **Escalabilidad:** Diseñado para adaptarse a diferentes entornos y escalas, desde pequeñas áreas forestales hasta vastas extensiones de terreno.
            
        - **Facilidad de uso:** Interfaz intuitiva y amigable que permite a los usuarios monitorear y gestionar el sistema con facilidad.
            
        - **Impacto ambiental:** Al detectar incendios de manera temprana, ayudamos a prevenir la propagación y minimizamos el impacto ambiental y económico de los incendios forestales. 
            
        📈 ¡Únete a la revolución de la seguridad ambiental y haz la diferencia hoy mismo! ¡No esperes más para proteger nuestro planeta con nuestra innovadora solución de detección de incendios!
            
        🔥 ¡No dejes que el fuego arruine nuestro futuro! 🔥

        #### Descripción del proyecto

        **Este proyecto se basa en un modelo de reconocimiento y segmentación de imágenes en tiempo real entrenado para detectar si hay fuego en la imagen, utiliza un algoritmo llamado YOLO que se centra en este tipo de modelos haciendo que sea mucho más veloz y preciso.**
    """)

# -- CONFIGURACION --
# Funcion para validar correo
def validar_correo(email):
    # Expresión regular para validar correo electrónico
    regex = r'^(([^<>()\[\]\\.,;:\s@\"]+(\.[^<>()\[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$'
    
    # Validar el correo con la expresión regular
    if re.match(regex, email):
        return True
    else:
        return False
# Funcion para mostrar la informacion del apartado 'Configuracion'
def tab_configuracion():

    st.subheader("Configuración: ", divider = "red")    
    
    # Solicitamos al usuario su correo electrónico
    email_usuario = st.text_input("Ingrese su correo electrónico para recibir la alerta de fuego: ")

    # Comprobamos que ha introducido un correo 
    if email_usuario:
        
        if validar_correo(email_usuario):
            
            # Eliminamos el valor del estado de sesion
            del state.email
            # Le aplicamos a este estado el valor del email introducido
            state.email = email_usuario
        else:
            st.warning("Debe introducir un correo electrónico válido", icon="⚠️")

    # Mostramos un selector de idioma al usuario
    lang = st.selectbox("Selecione un idioma:", options=list(LANGUAGES.keys()))

    # Comprobamos que ha seleccionado un idioma
    if lang:
        # Eliminamos el valor del estado de sesion
        del state.langs
        # Le aplicamos a este estado el valor del idioma elegido
        state.langs = lang

# -- DETECCION --

# Inicializamos la clase 'ProcesadorVideo'
class ProcesadorVideo:

    # Constructor de la clase
    def __init__(self):
        pass

    # Metodo para recibir y procesar cada frame del video
    def recv(self, frame):

        # Transformamos el frame de Streamlit a una imagen de numpy BGR
        img = frame.to_ndarray(format="bgr24")
        # Usamos el modelo YOLO para predecir el fuego en el frame
        resultados = model.predict(img, imgsz=640, conf=0.37)
        # Añadimos la segmentacion con la deteccion a la imagen
        anotaciones = resultados[0].plot()

        # Convertimos el array de numpy a frame de Streamlit con el mismo formato
        return av.VideoFrame.from_ndarray(anotaciones, format="bgr24")

# Funcion para mostrar la informacion del apartado 'Deteccion'
def tab_deteccion():

    st.subheader("Detección de Vídeo: ", divider = "red")    

    # Configuracion de WebRTC (Web Real-Time Communication)
    # Establecemos el servidor ICE (Interactive Connectivity Establishment)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Configuracion y creacion del flujo de video mediante WebRTC
    # - key: Clave unica para identificar el flujo de vídeo
    # - mode: Modo de WebRTC, en este caso, SENDRECV, que permite enviar y recibir datos
    # - rtc_configuration: Configuracion del servidor ICE para la conexion WebRTC
    # - media_stream_constraints: Restricciones para la transmisión de medios, en este caso solo se permite video
    # - video_processor_factory: Clase que procesara cada frame de vídeo, en este caso, ProcesadorVideo
    # - async_processing: Habilita el procesamiento asincrono para evitar bloquear la interfaz de usuario
    webrtc_ctx = webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=ProcesadorVideo,
        async_processing=True,
    )

# -- ALARMA --

# TTS (Text To Speech):

# Pasar de texto a audio
# Parametros:
# - text: Texto para pasar a audio
# - lang: Idioma con el que convertir el texto 
# - slow: Indicar la velocidad de reproduccion del audio
def text_to_speech(text, lang='en', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    return tts

# Función para reproducir el audio de manera automatica
def autoplay_audio(ruta_archivo: str):
    # Abrimos el archivo en modo lectura binaria
    with open(ruta_archivo, "rb") as archivo:
        # Leemos los datos y los guardamos en una variable
        data = archivo.read()
        # Codificamos los datos en base64 para insertarlo en HTML
        b64 = base64.b64encode(data).decode()
        # Codigo HTML para reproducir de forma automatica el audio
        md = f"""
            <audio controls autoplay="true" style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        
        # Mediante markdown lo introducimos en la página para que se ejecute
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

# Funcion para mostrar la informacion del apartado 'Alarma'
def tab_alarma():

    st.subheader("Alarma: ", divider = "red")  

    # Guardamos en una variable el estado actual del idioma
    lang = state.langs

    # Checkbox para indicar si enviar un email o no
    env_mail = st.checkbox('Enviar correo electrónico')

    # Comprobamos que si se ha seleccionado
    if env_mail:

        # Verificamos que el email tiene valores mediante su estado de sesion
        if not state.email:
             # Si se selecciona 'env_mail' sin tener un email mostramos un warning
             st.warning("Debe introducir un correo electrónico en la casilla de configuración", icon="⚠️")

    # Funcion para ejecutar el audio de alarma y envio de correo si se detecta fuego
    def audio_fuego():
        tts = text_to_speech(texto_audio[LANGUAGES[lang]], LANGUAGES[lang])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ruta_archivo:
            tts.save(ruta_archivo.name)
            autoplay_audio(ruta_archivo.name)
                        
        if env_mail:
    
            if state.email:
                try: 
                    enviar_correo(state.email)
                except:
                    st.error("Error al enviar correo electrónico", icon="🚨")
            else: 
                st.error("Debe introducir un correo electrónico", icon="🚨")

    # Funcion de streamlit que permite tomar capturas con la camara                 
    img_file_buffer = st.camera_input("")
    
    if img_file_buffer:
        # Leer img_file_buffer con CV2:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        resultado = model.predict(cv2_img, imgsz=640, conf=0.37)
        anotaciones = resultado[0].plot()
        anotaciones = cv2.cvtColor(anotaciones, cv2.COLOR_BGR2RGB)

        # Si detecta fuego llama a la funcion de audio 
        if len(resultado[0]) > 0:
            audio_fuego()
    
        st.image(anotaciones)

def no_detection():
    tts = text_to_speech(no_detecta[LANGUAGES[lang]], LANGUAGES[lang])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as ruta_archivo:
        tts.save(ruta_archivo.name)
        autoplay_audio(ruta_archivo.name)

# Tratado del texto
def procesar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar tildes
    texto = unidecode.unidecode(texto)
    
    # Tokenización
    tokens = word_tokenize(texto)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]

    # Comprobamos para cada token si coincide con uno de los apartados
    for token in tokens:
        if token in pre_titles:
            # Si coincide devolvemos el token
            return token
    
    # Unir tokens en un solo string
    processed_text = ' '.join(tokens)

    no_detection()

    # Si ningun token coincide devolvemos el texto procesado
    return processed_text

for title in tab_titles:
    pre_titles.append(procesar_texto(title))

# Contenido visible de la página
def main():
    
    # Logo
    st.image("Imgs/LogoFireDetectorDerecha.png", width=350)

    # Cabecera
    st.title("Fire Detector")
    st.write("By Alejandro Fernández & Andrés García")

    # Speech to text
    # Comprobamos si no tiene valor le asiganamos "Introduccion"    
    if 'mostrar' not in state:
        state.mostrar = "introduccion"

    # Creamos y centramos el boton de reconocimiento de voz
    col1, dol2, col3, col4 = st.columns(4)
    with col1:
        texto_audio = speech_to_text(language='es', start_prompt="Asistente Virtual", stop_prompt="Parar", use_container_width=True, just_once=True, key='STT')

    # Procesamos el texto captado
    if texto_audio:
        text = procesar_texto(texto_audio)

        # Compara los nombres de las paginas con el texto captado
        for title in pre_titles:
            if title in text:
                del state.mostrar
                state.mostrar = title

        st.write("Texto reconocido por el asistente:")
        st.text(texto_audio)
                   
    st.subheader("Menú: ", divider = "red")
   
    # Agregar los botones del menu en una fila horizontal
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        introduccion = st.button(tab_titles[0], use_container_width=True)
    with col2:
        configuracion = st.button(tab_titles[1], use_container_width=True)
    with col3:
        deteccion = st.button(tab_titles[2], use_container_width=True)
    with col4:
        alarma = st.button(tab_titles[3], use_container_width=True)

    # Si el boton es pulsado nos redirige a la ruta correspondiente
    if introduccion:
        del state.mostrar
        state.mostrar = pre_titles[0]
    elif configuracion:
        del state.mostrar
        state.mostrar = pre_titles[1]
    elif deteccion:
        del state.mostrar
        state.mostrar = pre_titles[2]
    elif alarma:
        del state.mostrar
        state.mostrar = pre_titles[3]

    # Segun el estado de la variable ejecutamos una funcion u otra
    if state.mostrar == pre_titles[0]:
        tab_introduccion()
    elif state.mostrar == pre_titles[1]:
        tab_configuracion()
    elif state.mostrar == pre_titles[2]:
        tab_deteccion()
    elif state.mostrar == pre_titles[3]:
        tab_alarma()
        
            
if __name__ == "__main__":
    main()
