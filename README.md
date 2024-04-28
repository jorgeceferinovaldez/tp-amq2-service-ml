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

La API SCO-SDSS17 (Star Classification Objects SDSS) clasifica objetes celestes dependiendo
de sus cáracteristicas especialmente del corrimento al rojo. El repositorio tiene los siguientes componenetes :

- El DAG **process_et_stellar_data** en Apache Airflow. Que realiza el  preprocesamiento, normalización y econding de los datos. Además de divide en entrenamiento y testeo los datos para guardarlos en s3-minio y ser usados posteriormente para modelar los datos. 
- El experimento en MLflow implementado con Optuna en el notebook **experiment_mlflow.ipynb** para hacer una selección de los mejores hiperparámetros. 
- El artefacto del modelo ganador del experimento llamado **model.pkl**.  
- Un endpoint creado con FastAPI para servir el modelo y hacer predicciones sobre datos nuevos. 

# Instalación 

1. Para desplegar el paquete de servicios en la carpeta raíz de este repositorio, ejecute:

```{bash}
docker compose --profile all up
```
2. Probar el correcto despliegue de los servicios 

- Apache Airflow: http://localhost:8080
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001 (ventana de administración de Buckets)
- API: http://localhost:8800/
- Documentación de la API: http://localhost:8800/docs , aquí podras ver la descripción de las variables de entrada si tienes dudas para usar la API. 

# Run 

 Para ejecutar una predicción nueva puedes hacerlo de varias maneras que se describiran a continuación cambiando las carácteristicas de ejemplo: 

- Python

```{python}
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
```{bash}
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