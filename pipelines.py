from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from transformers import BeautifulSoupParser
from pyspark.ml.util import keyword_only

class PipelineEngine(object):
    def __init__(self):
        self.evaluator = BinaryClassificationEvaluator()

    def fit(self, train):
        cv = CrossValidator()
        cv.setEstimator(self.pipeline)
        cv.setEvaluator(self.evaluator)
        cv.setEstimatorParamMaps(self.param_grid)
        cv.setNumFolds(3)
        return cv.fit(train)


class BaselinePipelineEngine(PipelineEngine):
    @keyword_only
    def __init__(self):
        self._build_get_stages()
        self.pipeline = Pipeline(stages=[self.bs_parser, self.tokenizer, self.hashing_tf, self.idf_model, self.lr])
        self.param_grid = self._build_param_grid()
        super(BaselinePipelineEngine, self).__init__()

    def _build_get_stages(self):
        self.bs_parser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
        self.tokenizer = Tokenizer(inputCol=self.bs_parser.getOutputCol(), outputCol="words")
        self.hashing_tf = HashingTF(inputCol=self.tokenizer.getOutputCol(), outputCol="raw_features")
        self.idf_model = IDF(inputCol=self.hashing_tf.getOutputCol(), outputCol="features")
        self.lr = LogisticRegression(maxIter=10, regParam=0.01)

    def _build_param_grid(self):
        param_grid_builder = ParamGridBuilder()
        param_grid_builder.addGrid(self.hashing_tf.numFeatures, [pow(2, 20)])
        param_grid_builder.addGrid(self.lr.regParam, [0.1, 0.01])
        return param_grid_builder.build()


class SentimentalPipeline(PipelineEngine):
    def __init__(self):
        super(SentimentalPipeline, self).__init__()
        # Todo
