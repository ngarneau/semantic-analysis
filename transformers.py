import nltk
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.ml.util import keyword_only
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from bs4 import BeautifulSoup
from nltk.tokenize import WhitespaceTokenizer


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
        self._setDefault(tokenizer=WhitespaceTokenizer())
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
            return tokenizer.tokenize(s)

        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = dataset[self.getInputCol()]
        return dataset.withColumn(out_col, udf(f, t)(in_col))

