import os

import click
from pyspark import SparkContext
from datasources import Datasources
from pipelines import BaselinePipelineEngine, SentimentalPipelineEngine


@click.command()
@click.option('--algorithm', default="baseline", help='Algorithm to run on dataset.')
@click.option('--evaluate/--fit', default=False, help='Flag to evaluate or run fit the model on the whole dataset')
@click.option('--sample/--no-sample', default=False, help='Only run on a sample to save time')
def app(algorithm, evaluate, sample):
    sc = SparkContext("local", "Pipeline", pyFiles=["datasources.py", "transformers.py", "pipelines.py"])
    dt = Datasources(sc)
    pipeline_engine = get_pipeline(algorithm)

    original_training_set = dt.get_original_training_set()
    original_test_set = dt.get_original_test_set()
    if sample:
        original_training_set = original_training_set.limit(10)
        original_test_set = original_test_set.limit(10)

    if evaluate:
        metrics = pipeline_engine.evaluate(original_training_set)
        print("Area under ROC: %s" % metrics.areaUnderROC)
    else:
        model = pipeline_engine.fit(original_training_set)
        prediction = model.transform(original_test_set)
        id_label = prediction.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
        save_to_file(id_label, "predictions.csv")


def get_pipeline(algorithm):
    if algorithm == "baseline":
        return BaselinePipelineEngine()
    elif algorithm == "sentimental":
        return SentimentalPipelineEngine()
    else:
        raise RuntimeError("You must specify an algorithm")


def save_to_file(rdd, path):
    try:
        rdd.saveAsTextFile(path)
    except OSError:
        pass


if __name__ == '__main__':
    app()
