# Especialización en Inteligencia Artificial FIUBA

# Trabajo Práctico Integrador

# Aprendizaje Máquina II
---

## Integrantes:
   - Josselyn Ordoñez 
   - Tatiana Arenas 
   - Jorge Valdez


# 1. Dataset seleccionado.

*  Empleamos el dataset analizado en Aprendizaje Máquina I:
    - Dataset: Stellar Classification Dataset - SDSS17
    

## API SCO-SDSS17 (Sloan Digital Sky Survey)

La API SCO-SDSS17 (Star Classification Objects SDSS) clasifica objetos celestes dependiendo de sus características, especialmente del corrimiento al rojo. El repositorio tiene los siguientes componentes:

- El DAG **process_et_stellar_data** en Apache Airflow realiza el preprocesamiento, normalización y codificación de los datos. Además, divide los datos en entrenamiento y prueba para guardarlos en s3-MinIO y ser utilizados posteriormente para modelar los datos.
- El experimento en MLflow implementado con Optuna en el notebook **experiment_mlflow.ipynb** para realizar una selección de los mejores hiperparámetros.
- El artefacto del modelo ganador del experimento llamado **model.pkl**.
- Un endpoint creado con FastAPI para servir el modelo y realizar predicciones sobre datos nuevos.

# Instalación

1. Para desplegar el paquete de servicios en la carpeta raíz de este repositorio, ejecute:

```bash
docker-compose --profile all up
```

2. Probar el correcto despliegue de los servicios:

- **Apache Airflow**: Herramienta para programar, monitorear y administrar flujos de trabajo de datos.
  - URL: http://localhost:8080
- **MLflow**: Plataforma de código abierto para gestionar el ciclo de vida completo del aprendizaje automático.
  - URL: http://localhost:5000
- **MinIO**: Servidor de almacenamiento de objetos de alto rendimiento y distribuido.
  - URL: http://localhost:9001
- **API**: Endpoint de la API que sirve el modelo y realiza predicciones sobre datos nuevos.
  - URL: http://localhost:8800/
- **Documentación de la API**: Documentación interactiva de la API, donde puedes ver la descripción de las variables de entrada si tienes dudas para usar la API.
  - URL: http://localhost:8800/docs

# Ejecución

Para realizar una predicción nueva, puedes hacerlo de varias maneras que se describirán a continuación cambiando las características de ejemplo:

- Python

```python
import requests
import json

# Definir la URL y los datos a enviar
url = 'http://localhost:8800/predict/'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "features": {
        "MJD": 58583,
        "alpha": 180,
        "cam_col": 3,
        "delta": 45,
        "fiber_ID": 456,
        "field_ID": 674,
        "g": 21.6,
        "i": 20.9,
        "obj_ID": 123456789,
        "plate": 2345,
        "r": 21.1,
        "redshift": 0.123,
        "run_ID": 756,
        "spec_obj_ID": 567890123,
        "u": 22.4,
        "z": 20.7
    }
}

# Realizar la solicitud POST
response = requests.post(url, headers=headers, data=json.dumps(data))

# Imprimir la respuesta del servidor
print(response.status_code)
print(response.json())
```

- Bash

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
        "MJD": 58583,
        "alpha": 180,
        "cam_col": 3,
        "delta": 45,
        "fiber_ID": 456,
        "field_ID": 674,
        "g": 21.6,
        "i": 20.9,
        "obj_ID": 123456789,
        "plate": 2345,
        "r": 21.1,
        "redshift": 0.123,
        "run_ID": 756,
        "spec_obj_ID": 567890123,
        "u": 22.4,
        "z": 20.7
    }
}'
```