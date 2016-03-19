import unittest

from pyspark import SparkContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, NGram
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from datasources import Datasources
from transformers import BeautifulSoupParser
from pyspark.ml.util import keyword_only


class PipelineEngine(object):
    def __init__(self):
        self.evaluator = BinaryClassificationEvaluator()
        self.pipeline = None
        self.param_grid = None

    def _build_stages(self):
        raise NotImplementedError()

    def _build_param_grid(self):
        raise NotImplementedError()

    def fit(self, train):
        """
        Train and return a model on the dataframe
        Args:
            train (Dataframe): 
        """
        cv = CrossValidator()
        cv.setEstimator(self.pipeline)
        cv.setEvaluator(self.evaluator)
        cv.setEstimatorParamMaps(self.param_grid)
        cv.setNumFolds(3)
        return cv.fit(train)

    def evaluate(self, train):
        train, test = train.randomSplit([0.6, 0.4], 1234)
        model = self.fit(train)
        prediction = model.transform(test)
        prediction_and_labels = prediction.map(lambda s: (s.prediction, s.label))
        return BinaryClassificationMetrics(prediction_and_labels)


class BaselinePipelineEngine(PipelineEngine):
    @keyword_only
    def __init__(self):
        super(BaselinePipelineEngine, self).__init__()
        self.hashing_tf_map = [pow(2, 20)]
        self.lr_map = [0.1, 0.01]
        self.stages = self._build_stages()
        self.pipeline = Pipeline(stages=[self.bs_parser, self.tokenizer, self.hashing_tf, self.idf_model, self.lr])
        self.param_grid = self._build_param_grid()

    def _build_stages(self):
        self.bs_parser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
        self.tokenizer = Tokenizer(inputCol=self.bs_parser.getOutputCol(), outputCol="words")
        self.hashing_tf = HashingTF(inputCol=self.tokenizer.getOutputCol(), outputCol="raw_features")
        self.idf_model = IDF(inputCol=self.hashing_tf.getOutputCol(), outputCol="features")
        self.lr = LogisticRegression(maxIter=10, regParam=0.01)
        return [self.bs_parser, self.tokenizer, self.hashing_tf, self.idf_model, self.lr]

    def _build_param_grid(self):
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(self.hashing_tf.numFeatures, self.hashing_tf_map)
        param_grid_builder.addGrid(self.lr.regParam, self.lr_map)
        return param_grid_builder.build()


class SentimentalPipelineEngine(PipelineEngine):
    def __init__(self):
        super(SentimentalPipelineEngine, self).__init__()
        self.ngram_map = [1, 2, 3]
        self.hashing_tf_map = [pow(2, 20)]
        self.stages = self._build_stages()
        self.pipeline = Pipeline(stages=self.stages)
        self.param_grid = self._build_param_grid()

    def _build_stages(self):
        self.bs_parser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
        self.tokenizer = Tokenizer(inputCol=self.bs_parser.getOutputCol(), outputCol="words")
        self.ngram = NGram(inputCol=self.tokenizer.getOutputCol(), outputCol="ngrams")
        self.hashing_tf = HashingTF(inputCol=self.ngram.getOutputCol(), outputCol="raw_features")
        self.idf_model = IDF(inputCol=self.hashing_tf.getOutputCol(), outputCol="features")
        self.lr = LogisticRegression(maxIter=10, regParam=0.01)
        return [self.bs_parser, self.tokenizer, self.ngram, self.hashing_tf, self.idf_model, self.lr]

    def _build_param_grid(self):
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(self.ngram.n, self.ngram_map)
        param_grid_builder.addGrid(self.hashing_tf.numFeatures, self.hashing_tf_map)
        return param_grid_builder.build()


class SentimentalPipelineTest(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext("local", "test_app")
        self.pipeline = SentimentalPipelineEngine()

    def tearDown(self):
        self.sc.stop()

    def test_build_stages(self):
        self.assertEqual(len(self.pipeline.stages), 6)

    def test_build_param_grid(self):
        param_grid = self.pipeline.param_grid
        self.assertEqual(len(param_grid), len(self.pipeline.ngram_map) * len(self.pipeline.hashing_tf_map))


class BaselinePipelineTest(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext("local", "test_app")
        self.pipeline = BaselinePipelineEngine()

    def tearDown(self):
        self.sc.stop()

    def test_build_stages(self):
        self.assertEqual(len(self.pipeline.stages), 5)

    def test_build_param_grid(self):
        param_grid = self.pipeline.param_grid
        self.assertEqual(len(param_grid), len(self.pipeline.lr_map) * len(self.pipeline.hashing_tf_map))


if __name__ == "__main__":
    unittest.main()
