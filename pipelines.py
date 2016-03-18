from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from transformers import BeautifulSoupParser
from pyspark.ml.util import keyword_only


class BaselinePipelineWrapper(object):
    @keyword_only
    def __init__(self):
        self.bs_parser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
        self.tokenizer = Tokenizer(inputCol=self.bs_parser.getOutputCol(), outputCol="words")
        self.hashing_tf = HashingTF(inputCol=self.tokenizer.getOutputCol(), outputCol="raw_features")
        self.idf_model = IDF(inputCol=self.hashing_tf.getOutputCol(), outputCol="features")
        self.lr = LogisticRegression(maxIter=10, regParam=0.01)
        self.pipeline = Pipeline(stages=[self.bs_parser, self.tokenizer, self.hashing_tf, self.idf_model, self.lr])
