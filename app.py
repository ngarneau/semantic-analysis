import click
import time
from pyspark import SparkContext, SQLContext
from pyspark.ml.tuning import CrossValidator
from datasources import Datasources
from pipelines import BaselinePipelineEngine, SentimentalPipelineEngine


@click.command()
@click.option('--algorithm', default="sentimental", help='Algorithm to run on dataset.')
@click.option('--train', default="data/labeledTrainData.tsv", help='Train dataset.')
@click.option('--test', default="data/testData.tsv", help='Test dataset.')
@click.option('--output', default="data/output", help='Place to put the results.')
@click.option('--evaluate/--fit', default=False, help='Flag to evaluate or run fit the model on the whole dataset')
@click.option('--sample/--no-sample', default=False, help='Only run on a sample to save time')
@click.option('--amazon', default="data/amazon", help='Train dataset.')
def app(algorithm, train, test, output, evaluate, sample, amazon):
    sc = SparkContext(appName="Pipeline")
    sql_context = SQLContext(sc)
    dt = Datasources(sc)
    pipeline_engine = get_pipeline(algorithm)

    if sample:
        original_training_set = dt.get_sample_training_set()
        original_test_set = dt.get_sample_test_set()
    else:
        # amazon_neg = sql_context.read.json(amazon + "_neg")
        # amazon_pos = sql_context.read.json(amazon + "_pos")
        original_training_set = dt.get_original_training_set(train)
        # original_training_set = original_training_set.unionAll(amazon_neg).unionAll(amazon_pos)
        original_test_set = dt.get_original_test_set(test)

    if evaluate:
        metrics = pipeline_engine.evaluate(original_training_set)
        metrics_str = "Area under ROC: %s\n" % metrics.areaUnderROC
        # metrics_str += "Params map: " + str(pipeline_engine.cv.metrics) + "\n"
        # metrics_str += "Metrics: " + str(pipeline_engine.cv.getEstimatorParamMaps()) + "\n"
        write_metrics_to_file(metrics_str, output + "_metrics")

        prediction = pipeline_engine.model.transform(original_test_set)
        id_label = prediction.rdd.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
        save_to_file(id_label, output)
    else:
        model = pipeline_engine.fit(original_training_set)
        prediction = model.transform(original_test_set)
        id_label = prediction.rdd.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
        save_to_file(id_label, output)


def write_metrics_to_file(metrics_str, output):
    file = open(output + "_" + str(time.time()) + ".txt", "w")
    file.write(metrics_str)
    file.close()


def get_pipeline(algorithm):
    if algorithm == "baseline":
        return BaselinePipelineEngine(cv=CrossValidator())
    elif algorithm == "sentimental":
        return SentimentalPipelineEngine(cv=CrossValidator())
    else:
        raise RuntimeError("You must specify an algorithm")


def save_to_file(rdd, path):
    try:
        rdd.saveAsTextFile(path + "_" + str(time.time()))
    except OSError:
        pass


if __name__ == '__main__':
    app()
