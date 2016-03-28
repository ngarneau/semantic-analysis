import click
from pyspark import SparkContext
from pyspark.ml.tuning import CrossValidator
from datasources import Datasources
from pipelines import BaselinePipelineEngine, SentimentalPipelineEngine


@click.command()
@click.option('--algorithm', default="sentimental", help='Algorithm to run on dataset.')
@click.option('--evaluate/--fit', default=False, help='Flag to evaluate or run fit the model on the whole dataset')
@click.option('--sample/--no-sample', default=False, help='Only run on a sample to save time')
def app(algorithm, evaluate, sample):
    sc = SparkContext("local", "Pipeline", pyFiles=["datasources.py", "transformers.py", "pipelines.py"])
    dt = Datasources(sc)
    pipeline_engine = get_pipeline(algorithm)

    if sample:
        original_training_set = dt.get_sample_training_set()
        original_test_set = dt.get_sample_test_set()
    else:
        original_training_set = dt.get_original_training_set("s3://ift-7025/labeledTrainData.tsv")
        original_test_set = dt.get_original_test_set("s3://ift-7025/testData.tsv")

    if evaluate:
        metrics = pipeline_engine.evaluate(original_training_set)
        print("Area under ROC: %s" % metrics.areaUnderROC)
    else:
        model = pipeline_engine.fit(original_training_set)
        prediction = model.transform(original_test_set)
        id_label = prediction.rdd.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
        save_to_file(id_label, "predictions.csv")


def get_pipeline(algorithm):
    if algorithm == "baseline":
        return BaselinePipelineEngine(cv=CrossValidator())
    elif algorithm == "sentimental":
        return SentimentalPipelineEngine(cv=CrossValidator())
    else:
        raise RuntimeError("You must specify an algorithm")


def save_to_file(rdd, path):
    try:
        rdd.saveAsTextFile(path)
    except OSError:
        pass


if __name__ == '__main__':
    app()
