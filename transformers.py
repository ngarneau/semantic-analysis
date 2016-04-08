from pyspark.mllib.linalg import Vectors, Vector

import nltk
import unittest
from pyspark import SparkContext, SQLContext
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, Param
from pyspark.ml.util import keyword_only
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, DataType, DataTypeSingleton, DoubleType
from bs4 import BeautifulSoup
from nltk.tokenize import WhitespaceTokenizer, TweetTokenizer
from nltk.stem import *
from nltk.sentiment.util import mark_negation
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class BeautifulSoupParser(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(BeautifulSoupParser, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        def f(s):
            return BeautifulSoup(s, 'html.parser').get_text()

        t = StringType()
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


class Tokenizzzer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, tokenizer=None):
        super(Tokenizzzer, self).__init__()
        self.tokenizer = Param(self, "tokenizer", "")
        self._setDefault(tokenizer=TweetTokenizer())
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, tokenizer=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def setTokenizer(self, value):
        self._paramMap[self.tokenizer] = value
        return self

    @keyword_only
    def getTokenizer(self):
        return self.getOrDefault(self.tokenizer)

    def _transform(self, dataset):
        tokenizer = self.getTokenizer()

        def f(s):
            return mark_negation(tokenizer.tokenize(s))

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


class PorterStemmerTransformer(Transformer, HasInputCol, HasOutputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(PorterStemmerTransformer, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        stemmer = PorterStemmer()

        def f(l):
            return [stemmer.stem(s) for s in l]

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))


class VaderPolarizer(Transformer, HasInputCol):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(VaderPolarizer, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        analyzer = SentimentIntensityAnalyzer()

        def polarize(text):
            polarized = analyzer.polarity_scores(text)
            if polarized['pos'] > polarized['neg']:
                return 1.0
            else:
                return 0.0

        t = DoubleType()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn("polarized", udf(polarize, t)(in_col))


class SparkTest(unittest.TestCase):
    def setUp(self):
        self.sc = SparkContext("local", "test_app")

    def tearDown(self):
        self.sc.stop()


class VaderPolarizerTest(SparkTest):
    def test_fit(self):
        data = self._get_data()
        transformer = VaderPolarizer(inputCol='text')
        transformed_data = transformer._transform(data)
        self.assertIn('polarized', transformed_data.columns)

    def _get_data(self):
        sql_context = SQLContext(self.sc)
        l = [
            (
            "I dont know why people think this is such a bad movie.",
            Vectors.sparse(3, {1: 1.0, 2: 1.0, 3: 1.0})
            ),
        ]
        return sql_context.createDataFrame(l, ['text', 'features'])



if __name__ == "__main__":
    unittest.main()
