#from hate.pipeline.training_pipeline import TrainPipeline
#if __name__ == "__main__":
#    TrainPipeline().run_pipeline()

from hate.pipeline.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":
    PredictionPipeline().run_pipeline(text="humans are bad")