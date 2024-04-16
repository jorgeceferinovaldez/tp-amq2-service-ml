import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Stellar Classification Data

This DAG extracts information from the Stellar Classification Dataset - SDSS17, which contains data on stellar objects observed by the Sloan Digital Sky Survey. 
It preprocesses the data by encoding categorical variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for testing. 
The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Jorge Ceferino Valdez",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_stellar_data",
    description="ETL process for stellar data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Stellar Classification"],
    default_args=default_args,
    catchup=False,
)
def process_etl_star_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from the "star_classification.csv" file.
        """
        import awswrangler as wr
        from airflow.models import Variable

        import requests
        import pandas as pd

        # Paso la url del archivo compartido en Google Drive
        file_url = 'https://drive.google.com/uc?id=17iWMy-9jCws_EjG0qKQIfnvRp086q73W'

        # Descargar el archivo desde la URL
        response = requests.get(file_url)

        # Guardar el contenido descargado en un archivo local
        with open('star_classification.csv', 'wb') as f:
            f.write(response.content)

        # Cargar datos desde el archivo CSV descargado
        dataframe = pd.read_csv('star_classification.csv')

        target_col = Variable.get("target_col_star")


        # Assuming you have similar preprocessing steps as in the original code
        # Replace this with your preprocessing steps if necessary

        wr.s3.to_csv(df=dataframe,
                     path="s3://data/raw/star.csv",
                     index=False) # Guardar el archivo en S3 o en el bucket de S3 que en este caso es Minio

    @task.virtualenv(
        task_id="preprocess_data",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def preprocess_data():
        """
        Preprocess the data by encoding categorical variables into numerical ones.
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        
        from airflow.models import Variable
        import pandas as pd
        import numpy as np

        from sklearn.preprocessing import LabelEncoder

        # Load data
        data_original_path = "s3://data/raw/star.csv"
        data_end_path = "s3://data/preprocessed/star_preprocess.csv"
        df = wr.s3.read_csv(data_original_path) # Cargar el archivo de S3 como dataframe de pandas

        # Convertimos la columna 'class' a valores numéricos
        # Definir una función lambda para mapear los valores a números
        df_preprocess = df.copy()

        le = LabelEncoder()

        df_preprocess['class']=le.fit_transform(df_preprocess['class'])

        # convertimos la columna 'cam_col' a valores numéricos
        df_preprocess['cam_col']=le.fit_transform(df_preprocess['cam_col'])

        # Elimino la columna rerun_ID ya que posee un solo valor
        df.drop(columns=['rerun_ID'], inplace=True)
        df_preprocess.drop(columns=['rerun_ID'], inplace=True)

        
        categories_list = ["cam_col"]
        # Save data
        wr.s3.to_csv(df=df_preprocess, 
                     path=data_end_path, 
                     index=False) # Guardo en Minio S3
        
        # Guardar la informacion del dataset en el bucket de Minio
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data_star.json')
            result = client.get_object(Bucket='data', Key='data_info/data_star.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = Variable.get("target_col_star")
        dataset_log = df.drop(columns=target_col)
        dataset_preprocess_log = df_preprocess.drop(columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_dummy'] = dataset_preprocess_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categories_list
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_preprocess'] = {k: str(v) for k, v in dataset_preprocess_log.dtypes
                                                                                                 .to_dict()                                                                                               .items()}

        category_preprocess_dict = {}
        for category in categories_list:
            category_preprocess_dict[category] = np.sort(dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_preprocess_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data_star.json',
            Body=data_string
        )        
        
        # MLFlow tracking
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Star Classification")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Star classification"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(df,
                                                 source="https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17",
                                                 targets=target_col,
                                                 name="star_data_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(df_preprocess,
                                                         source="https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17",
                                                         targets=target_col,
                                                         name="star_data_preprocessed")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")
   
    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Split the dataset into training and testing sets.
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path): # Guardar el archivo en S3
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        # Load data
        data_path = "s3://data/preprocessed/star_preprocess.csv"
        dataset = wr.s3.read_csv(data_path)

        test_size = Variable.get("test_size_star") # Leo el tamaño del test desde las variables de Airflow
        target_col = Variable.get("target_col_star") # Leo la columna target desde las variables de Airflow

        X = dataset.drop(columns=target_col)
        y = dataset[target_col]

        # Dividimos el dataset en training y testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Guardamos los datasets en S3
        save_to_csv(X_train, "s3://data/final/train/star_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/star_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/star_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/star_y_test.csv")

    @task.virtualenv(
        task_id="standardize_data",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def standardize_data():
        """
        Standardize the data by scaling the numerical features.
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        # Cargamos los datos de entrenamiento y testing desde S3
        X_train_path = "s3://data/final/train/star_X_train.csv"
        X_test_path = "s3://data/final/test/star_X_test.csv"
        
        y_train_path = "s3://data/final/train/star_y_train.csv"
        y_test_path = "s3://data/final/test/star_y_test.csv"

        X_train = wr.s3.read_csv(X_train_path)
        X_test = wr.s3.read_csv(X_test_path)
        y_train = wr.s3.read_csv(y_train_path)
        y_test = wr.s3.read_csv(y_test_path)

        # Standardize the data
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convertimos los arrays en DataFrames
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/star_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/star_X_test.csv")

        # Guardar la información del Standard Scaler en el bucket de Minio
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data_star.json')
            result = client.get_object(Bucket='data', Key='data_info/data_star.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = scaler.mean_.tolist()
        data_dict['standard_scaler_std'] = scaler.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data_star.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Star Classification")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", scaler.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", scaler.mean_)
            mlflow.log_param("Standard Scaler scale values", scaler.scale_)

    get_data() >> preprocess_data() >> split_dataset() >> standardize_data() # This is necessary to add the task to the DAG


dag = process_etl_star_data()