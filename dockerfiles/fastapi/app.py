import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0


    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data_star.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data_star.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        #data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        #data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data_star.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "star_class_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the star classification model.

    This class defines the input fields required by the star classification model along with their descriptions
    and validation constraints.

    :param obj_ID: Object identifier, a unique value that identifies the object in the image catalog used by CAS.
    :param alpha: Right Ascension angle (in epoch J2000) as a floating point number.
    :param delta: Declination angle (in epoch J2000) as a floating point number.
    :param u: Ultraviolet filter in the photometric system as a floating point number.
    :param g: Green filter in the photometric system as a floating point number.
    :param r: Red filter in the photometric system as a floating point number.
    :param i: Near-infrared filter in the photometric system as a floating point number.
    :param z: Infrared filter in the photometric system as a floating point number.
    :param run_ID: Run number used to identify the specific analysis as an integer.
    :param cam_col: Camera column to identify the scan line within the run. Integer values indicating the camera column position.
    :param field_ID: Field number to identify each field as an integer.
    :param spec_obj_ID: Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID should share the output class) as a floating point number.
    :param redshift: Redshift value based on the wavelength increase as a floating point number.
    :param plate: Plate ID, identifies each plate in SDSS as an integer.
    :param MJD: Modified Julian Date, used to indicate when a particular SDSS data was taken as an integer.
    :param fiber_ID: Fiber ID that identifies the fiber that directed light to the focal plane in each observation as an integer.

    """

    obj_ID: float = Field(
    description="Object identifier, a unique value that identifies the object in the image catalog used by CAS.",
    ge=0.0
    )
    alpha: float = Field(
        description="Right Ascension angle (in epoch J2000).",
        ge=0.0,
        le=360.0
    )
    delta: float = Field(
        description="Declination angle (in epoch J2000).",
        ge=-90.0,
        le=90.0
    )
    u: float = Field(
        description="Ultraviolet filter in the photometric system.",
        ge=0.0
    )
    g: float = Field(
        description="Green filter in the photometric system.",
        ge=0.0
    )
    r: float = Field(
        description="Red filter in the photometric system.",
        ge=0.0
    )
    i: float = Field(
        description="Near-infrared filter in the photometric system.",
        ge=0.0
    )
    z: float = Field(
        description="Infrared filter in the photometric system.",
        ge=0.0
    )
    run_ID: int = Field(
        description="Run number used to identify the specific analysis.",
        ge=0
    )
    cam_col: int = Field(
        description="Camera column to identify the scan line within the run.",
        ge=1,
        le=6
    )
    field_ID: int = Field(
        description="Field number to identify each field.",
        ge=0
    )
    spec_obj_ID: float = Field(
        description="Unique ID used for optical spectroscopic objects.",
        ge=0.0
    )
    redshift: float = Field(
        description="Redshift value based on the wavelength increase.",
        ge=0.0
    )
    plate: int = Field(
        description="Plate ID, identifies each plate in SDSS.",
        ge=0
    )
    MJD: int = Field(
        description="Modified Julian Date, used to indicate when a particular SDSS data was taken.",
        ge=0
    )
    fiber_ID: int = Field(
        description="Fiber ID that identifies the fiber that directed light to the focal plane in each observation.",
        ge=0
    )


    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "obj_ID": 123456789.0,
                    "alpha": 180.0,
                    "delta": 45.0,
                    "u": 22.4,
                    "g": 21.6,
                    "r": 21.1,
                    "i": 20.9,
                    "z": 20.7,
                    "run_ID": 756,
                    "cam_col": 3,
                    "field_ID": 674,
                    "spec_obj_ID": 567890123.0,
                    "redshift": 0.123,
                    "plate": 2345,
                    "MJD": 58583,
                    "fiber_ID": 456
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the star classification model.

    This class defines the output fields returned by the star classification model along with their descriptions
    and possible values.

    :param str_output: Output of the model in string form. Can be "Healthy patient" or "Heart disease detected".
    """

    int_output: int = Field(
        description="Output of the model. 0 for Galaxy, 1 for OSO and 2 for Star",
    )
    str_output: Literal["Galaxy", "OSO","Star"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 2,
                    "str_output": "Star",
                }
            ]
        }
    }



# Load the model before start
model, version_model, data_dict = load_model("star_class_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Star Classifier API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Star Classifier API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for classifying stars.

    This endpoint receives features related to spectral features and predicts whether it is a galaxy, OSO or star using a trained model. 
    It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Scale the data using standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Star"
    if prediction==0:
        str_pred = "Galaxy"
    elif prediction==1:
        str_pred = "OSO"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)

