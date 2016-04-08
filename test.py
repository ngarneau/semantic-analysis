from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext("local", "Pipeline")
sqlContext = SQLContext(sc)
dataset = sqlContext.createDataFrame(
    [(Vectors.dense([0.0]), 0.0),
     (Vectors.dense([0.4]), 1.0),
     (Vectors.dense([0.5]), 0.0),
     (Vectors.dense([0.6]), 1.0),
     (Vectors.dense([1.0]), 1.0)] * 10,
    ["features", "label"])
lr = LogisticRegression()
grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
cvModel = cv.fit(dataset)
print(cv.metrics)


Params map: [ 2.80026035  2.77896443  2.52157438  2.77129878  2.68407165  2.29883198]
Metrics: [
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): <nltk.tokenize.casual.TweetTokenizer object at 0x105452128>, Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 1, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576},
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): <nltk.tokenize.casual.TweetTokenizer object at 0x105452128>, Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 2, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576},
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): <nltk.tokenize.casual.TweetTokenizer object at 0x105452128>, Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 3, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576},
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): WhitespaceTokenizer(pattern='\\s+', gaps=True, discard_empty=True, flags=56), Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 1, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576},
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): WhitespaceTokenizer(pattern='\\s+', gaps=True, discard_empty=True, flags=56), Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 2, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576},
    {Param(parent='Tokenizzzer_47c4ad546cc0174c5bf9', name='tokenizer', doc=''): WhitespaceTokenizer(pattern='\\s+', gaps=True, discard_empty=True, flags=56), Param(parent='NGram_499e8ba0c19d556e369c', name='n', doc='number of elements per n-gram (>=1)'): 3, Param(parent='HashingTF_489387460f2680f2d6f8', name='numFeatures', doc='number of features.'): 1048576}]

Params map: [ 2.80151508]
Metrics: [{Param(parent='NGram_4b338738c901a197c6db', name='n', doc='number of elements per n-gram (>=1)'): 1, Param(parent='Tokenizzzer_4ef1808e516e9e78d783', name='tokenizer', doc=''): <nltk.tokenize.casual.TweetTokenizer object at 0x1054524a8>, Param(parent='HashingTF_4623a4e0650badd052c3', name='numFeatures', doc='number of features.'): 1048576}]