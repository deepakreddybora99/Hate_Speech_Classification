import os
import sys
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException

from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):

        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        self.data_ingestion_config = data_ingestion_config 



    def unzip_and_clean(self):

        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e
    

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:

        """
        Method Name :   initiate_data_ingestion
        Description :   This function initiates a data ingestion steps
        Output      :   Returns data ingestion artifact
        On Failure  :   Write an exception log and then raise an exception
        """


        try:
            raw_data_file_path = self.unzip_and_clean()

            logging.info("Unzipped file and split into train and valid")

            data_ingestion_artifacts = DataIngestionArtifacts(raw_data_file_path=raw_data_file_path)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e