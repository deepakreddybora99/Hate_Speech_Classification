import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.constants import *
from hate.model import ModelArchitecture
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts 



class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def get_best_model(self) -> str:
        try:
            logging.info("Entered the get_best_model method of Model Evaluation class")
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info("Exited the get_best_model method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self):
        try:
            logging.info("Entering into the evaluate function of Model Evaluation class")
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str)
            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"Test accuracy: {accuracy}")

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = [1 if prediction[0] >= 0.5 else 0 for prediction in lstm_prediction]
            logging.info(f"Confusion Matrix: {confusion_matrix(y_test, res)}")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logging.info("Initiate Model Evaluation")
        try:
            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            trained_model_accuracy = self.evaluate()
            best_model_path = self.get_best_model()

            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
            else:
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate()
                is_model_accepted = trained_model_accuracy > best_model_accuracy

            # Save model and tokenizer if accepted
            if is_model_accepted:
                predict_model_path = os.path.join("artifacts", "PredictModel", self.model_evaluation_config.MODEL_NAME)
                os.makedirs(os.path.dirname(predict_model_path), exist_ok=True)
                trained_model.save(predict_model_path)

                with open(os.path.join("artifacts", "PredictModel", "tokenizer.pickle"), "wb") as handle:
                    pickle.dump(tokenizer, handle)

            return ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
        except Exception as e:
            raise CustomException(e, sys) from e