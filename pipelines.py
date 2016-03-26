import unittest
import mock
from unittest.mock import MagicMock

from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, NGram
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.linalg import Vectors

from datasources import Datasources
from transformers import BeautifulSoupParser, Tokenizzzer
from pyspark.ml.util import keyword_only
from nltk.tokenize import TweetTokenizer, WhitespaceTokenizer


class PipelineEngine(object):
    def __init__(self, cv):
        self.cv = cv
        self.evaluator = BinaryClassificationEvaluator()
        self.pipeline = None
        self.param_grid = None
        self.model = None

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
        self.cv.setEstimator(self.pipeline)
        self.cv.setEvaluator(self.evaluator)
        self.cv.setEstimatorParamMaps(self.param_grid)
        self.model = self.cv.fit(train)
        return self.model

    def evaluate(self, train):
        train, test = train.randomSplit([0.6, 0.4], 1234)
        model = self.fit(train)
        prediction = model.transform(test)
        prediction_and_labels = prediction.rdd.map(lambda s: (s.prediction, s.label))
        print("Params map: " + str(self.cv.metrics))
        print("Metrics: " + str(self.cv.getEstimatorParamMaps()))
        return BinaryClassificationMetrics(prediction_and_labels)


class BaselinePipelineEngine(PipelineEngine):
    @keyword_only
    def __init__(self, cv):
        super(BaselinePipelineEngine, self).__init__(cv)
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
    def __init__(self, cv):
        super(SentimentalPipelineEngine, self).__init__(cv)
        self.tokenizer_map = [TweetTokenizer(), WhitespaceTokenizer()]
        self.ngram_map = [1, 2, 3]
        self.hashing_tf_map = [pow(2, 20)]
        self.stages = self._build_stages()
        self.pipeline = Pipeline(stages=self.stages)
        self.param_grid = self._build_param_grid()

    def _build_stages(self):
        self.bs_parser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
        self.tokenizer = Tokenizzzer(inputCol=self.bs_parser.getOutputCol(), outputCol="words")
        self.ngram = NGram(inputCol=self.tokenizer.getOutputCol(), outputCol="ngrams")
        self.hashing_tf = HashingTF(inputCol=self.ngram.getOutputCol(), outputCol="raw_features")
        self.idf_model = IDF(inputCol=self.hashing_tf.getOutputCol(), outputCol="features")
        self.string_index = StringIndexer(inputCol="label", outputCol="indexed")
        self.random_forest = RandomForestClassifier(
            featuresCol=self.idf_model.getOutputCol(),
            labelCol=self.string_index.getOutputCol()
        )
        return [self.bs_parser, self.tokenizer, self.ngram, self.hashing_tf, self.idf_model, self.string_index, self.random_forest]

    def _build_param_grid(self):
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(self.tokenizer.tokenizer, self.tokenizer_map)
        param_grid_builder.addGrid(self.ngram.n, self.ngram_map)
        param_grid_builder.addGrid(self.hashing_tf.numFeatures, self.hashing_tf_map)
        return param_grid_builder.build()


class TestPipeline(PipelineEngine):
    def __init__(self, cv):
        super(TestPipeline, self).__init__(cv)
        self.lr_map = [0.01]
        self.stages = self._build_stages()
        self.pipeline = Pipeline(stages=[self.lr])
        self.param_grid = self._build_param_grid()

    def _build_stages(self):
        self.lr = LogisticRegression(maxIter=10, regParam=0.01)
        return [self.lr]

    def _build_param_grid(self):
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(self.lr.regParam, self.lr_map)
        return param_grid_builder.build()


class SparkTest(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext("local", "test_app")

    def tearDown(self):
        self.sc.stop()


class SentimentalPipelineTest(SparkTest):
    def test_build_stages(self):
        self.pipeline = SentimentalPipelineEngine(cv=CrossValidator())
        self.assertEqual(len(self.pipeline.stages), 6)

    def test_build_param_grid(self):
        self.pipeline = SentimentalPipelineEngine(cv=CrossValidator())
        param_grid = self.pipeline.param_grid
        self.assertEqual(len(param_grid),
                         len(self.pipeline.ngram_map) * len(self.pipeline.hashing_tf_map) * len(self.pipeline.tokenizer_map)
                         )


class BaselinePipelineTest(SparkTest):
    def test_build_stages(self):
        self.pipeline = BaselinePipelineEngine(cv=CrossValidator())
        self.assertEqual(len(self.pipeline.stages), 5)

    def test_build_param_grid(self):
        self.pipeline = BaselinePipelineEngine(cv=CrossValidator())
        param_grid = self.pipeline.param_grid
        self.assertEqual(len(param_grid), len(self.pipeline.lr_map) * len(self.pipeline.hashing_tf_map))


class PipelineEngineTest(SparkTest):
    @mock.patch('pyspark.ml.tuning.CrossValidator')
    def test_fit(self, mock_cv):
        mock_cv.fit.return_value = PipelineModel(stages=[])
        test_pipeline = TestPipeline(mock_cv)
        train = self._get_train_data()
        test_pipeline.fit(train)
        mock_cv.fit.assert_called_with(train)

    def _get_train_data(self):
        sql_context = SQLContext(self.sc)
        l = [
            (1, Vectors.dense([1, 2, 3]), 1.0),
            (2, Vectors.dense([1, 2, 3]), 0.0),
            (3, Vectors.dense([1, 2, 3]), 1.0),
            (4, Vectors.dense([1, 2, 3]), 0.0),
        ]
        return sql_context.createDataFrame(l, ['id', 'features', 'label'])


if __name__ == "__main__":
    unittest.main()
