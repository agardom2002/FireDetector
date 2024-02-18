# Fire Detector
***Proyecto realizado por [Alejandro Fernández Romero](https://github.com/AlexFdezRomero) y [Andrés García Domínguez](https://github.com/agardom2002).***

## Índice
[1. Justificación y descripción del proyecto.](#id1)

[2. Obtención de datos.](#id2)

[3. Limpieza de datos.](#id3) 

[4. Exploración y visualización.](#id4)

[5. Preparación de los datos para los algoritmos de Machine Learning.](#id5)

[6. Entrenamiento del modelo y comprobación del rendimiento.](#id6)

[7. NPL](#id7)

[8. Aplicación web.](#id8)

[9. Conclusiones.](#id9)

## 1. Justificación y descripción del proyecto.<a name="id1"></a>

🔥 ¡Atención a todos los amantes de la tecnología y la seguridad! 🔥 ¿Estás buscando una manera innovadora de proteger nuestros preciosos recursos naturales? ¡Entonces, este es el proyecto que estabas esperando!

🚒 Presentamos nuestro increíble Trabajo de Fin de Máster en Inteligencia Artificial y Big Data: ¡La solución definitiva para la detección de incendios en áreas naturales!

🌳 En un mundo donde los incendios forestales representan una seria amenaza para el medio ambiente y la seguridad pública, ¡nuestro proyecto brinda una solución inteligente y efectiva!

💡 Con la combinación de algoritmos de Inteligencia Artificial y análisis de Big Data, nuestro sistema puede detectar la presencia de fuego en áreas naturales de manera precisa y oportuna. ¡No más preocupaciones por incendios no detectados!

✅ ¿Por qué elegir nuestra solución?

- Eficiencia: Nuestro sistema utiliza algoritmos avanzados para analizar imágenes y datos en tiempo real, lo que permite una detección temprana y rápida de incendios.
- Precisión: Gracias a la inteligencia artificial, nuestro sistema reduce al mínimo los falsos positivos, garantizando una detección confiable.
- Escalabilidad: Diseñado para adaptarse a diferentes entornos y escalas, desde pequeñas áreas forestales hasta vastas extensiones de terreno.
- Facilidad de uso: Interfaz intuitiva y amigable que permite a los usuarios monitorear y gestionar el sistema con facilidad.
- Impacto ambiental: Al detectar incendios de manera temprana, ayudamos a prevenir la propagación y minimizamos el impacto ambiental y económico de los incendios forestales.
📈 ¡Únete a la revolución de la seguridad ambiental y haz la diferencia hoy mismo! ¡No esperes más para proteger nuestro planeta con nuestra innovadora solución de detección de incendios!

🔥 ¡No dejes que el fuego arruine nuestro futuro! 🔥

Este proyecto se basa en un modelo de reconocimiento y segmentación de imágenes en tiempo real entrenado para detectar si hay fuego en la imagen, utiliza un algoritmo llamado YOLO que se centra en este tipo de modelos
haciendo que sea mucho más veloz y preciso (Enlace a su [GitHub](https://github.com/ultralytics/ultralytics)).

<img src="https://drive.google.com/uc?id=1CURd9SyhdrQgs0kD-ZFQW8m44clrS6bJ" height="600px">

***Imagen obtenida de la predicción del modelo entrenado.***

## 2. Obtención de datos.<a name="id2"></a>

Al utilizar el algoritmo YOLOv8, necesitamos tanto las imágenes para entrenar el modelo cómo la segmentación de estas imágenes para indicar cuál es el target, en este caso el fuego.
Para ello, vamos a utilizar un dataset obtenido desde la página [Roboflow](https://universe.roboflow.com/aj-garcia-736tc/fire-dataset-for-yolov8).

<img src="https://drive.google.com/uc?id=1CCq5_j7wUpI4lIWwtwRNHIPj7MLmnPVK" height="150px">

## 3. Limpieza de datos. <a name="id3"></a>
## 4. Exploración y visualización.<a name="id4"></a>
## 5. Preparación de los datos para los algoritmos de Machine Learning.<a name="id5"></a>

Si queremos segmentar imágenes manualmente para añadirlas al dataset podemos instalar en nuestro equipo **labelme**. En este caso lo hemos instalado y utilizado con Visual Studio Code.

`pip install labelme`

Una vez instalado, se puede iniciar escribiendo **labelme** en la terminal. 
Indicamos la carpeta donde se encuentran las imágenes a segmentar. El sigiente paso es mediante el ratón, indicar la zona del objetivo (target) y etiquetarlo. 
Con cada imagen, se genera un archivo .json que indica las coordenadas de la segmentación del target en la imagen.

<img src="https://drive.google.com/uc?id=1j5LO06FLwN3qexNPkma1XLXLHwTlQCou" height="300px">

Cuando se hayan segmentado y etiquetado todas las imágenes tenemos que instalar **labelme2yolo** para transformar los datos para el algoritmo.

`pip install labelme2yolo`

Tras instalarlo debemos ejecutar el siguiente comando para preparar los datos:

`labelme2yolo --json_dir "Ruta de la carpeta con las imágenes"`

Se puede observar que al ejecutar la transformación se genera una carpeta que prepara el dataset para el entrenamiento, separa por un lado las imágenes de los archivos .json asociados. Además,
crea un archivo **dataset.yaml** que será el que utilizaremos para realizar el entrenamiento.

<img src="https://drive.google.com/uc?id=1qTqUW3hMEv5jayhPxI3plWtiEEChNsSe" height="250px">

## 6. Entrenamiento del modelo y comprobación del rendimiento.<a name="id6"></a>

Enlace al entrenamiento en un documento de [Google Colab](https://colab.research.google.com/drive/1mmFQI4K9Ic9whAI8TFCMuOtnLl3uUM4S?usp=sharing).

### **Entrenamiento del modelo YOLOv8**

Instalación de los paquetes necesarios:

* **ultralytics:** Para obtener y entrenar el modelo
* **roboflow:** Para descargar el dataset de imágenes para el entrenamiento

```
!pip install ultralytics==8.0.196
!pip install roboflow
```
Importamos las librerías necesarias tanto para descargar el dataset como para entrenar el modelo

```
import ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
```
Descargamos el dataset de imágenes. Para ello usamos la API proporcionada de Roboflow

```
rf = Roboflow(api_key="MAiCeSuy58yjlg2ma4QK")
project = rf.workspace("-jwzpw").project("continuous_fire")
dataset = project.version(6).download("yolov8")
```
Comando para realizar el entrenamiento del modelo

Los diferentes parámetros que se usan son :

- **task**: Sirve para indicar la tarea a realizar.
  - detect: Consiste en detectar objetos en una imagen o video, dibujando cuadros a su alrededor calificándolos según sus características. Puede detectar varios objetos a la vez.
  - segment: Segmenta una imagen en diferentes regiones basándose en el contenido de la imagen. A cada region se la asigna una etiqueta. En diferencia con *detect*, no es un cuadro, es la silueta del objeto.
  - classify: Clasifica una imagen en diferentes categorías basándose en su contenido.
  - pose: Detecta los puntos clave de un objeto en un fotograma y se utilizan
  para seguir el movimiento o estimar la pose.

- **mode**:
  - train: Afinar el modelo en un conjunto de datos personalizados o precargados. El proceso consiste en optimizar los parámetros para una mayor
  precisión.
  - val: Se utiliza para un modelo una vez entrenado. Evalúa su precisión y
  rendimiento, permitiendo ajustar los hiperparámetros para mejorar su rendimiento.
  - predict: Se carga el modelo entrenado y se le proporcionan nuevas imágenes o videos para ver su funcionamiento.
  - export: Permite exportar el modelo a un formato que pueda utilizarse para su
  despliegue.
  - track: Modo seguimiento. Con el modelo entrenado, se le proporciona un flujo de vídeo en directo para seguir objetos en tiempo real.
  - benchmark: Su utiliza para perfilar la velocidad y precisión de varios formatos de exportación. Con información como; el tamaño del formato exportado, las diferentes métricas y el tiempo de inferencia por imagen, podemos elegir el formato más óptimo.

- **model**: Modelo a utilizar, en este caso *yolov8s.pt*.
Para yolov8 hay diferentes variantes; **n** (nano), **s** (small), **m** (medium), **l** (large) y **x** (extra large).

- **data**: Ruta donde se encuentra el archivo **.yaml** que indica los diferentes directorios de entrenamiento y validación.

- **epochs**: Establecer el número de iteraciones de los datos de entrenamiento.

- **imgsz**: Especificar el tamaño de las imágenes.

- **plots**: Indicar que se generen gráficas para evaluar el rendimiento del entrenamiento.
```
!yolo task=detect mode=train model=yolov8s.pt data=/content/continuous_fire-6/data.yaml epochs=80 imgsz=640 plots=True
```
Forma para descargar la carpeta runs de forma local en nuestro equipo

```
from google.colab import files
files.download('/content/runs')
```
Montar nuestro Google Drive
```
from google.colab import drive
drive.mount('/content/drive')
```
Copiamos la carpeta runs a un directorio de nuestro Google Drive
```
!cp -r /content/runs /content/drive/MyDrive/runs
```

## 7. NPL<a name="id7"></a>
## 8. Aplicación web.<a name="id8"></a>
## 9. Conclusiones.<a name="id9"></a>
