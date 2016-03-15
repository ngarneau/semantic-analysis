from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from src.transformers import BeautifulSoupParser

sc = SparkContext("local", "Pipeline")
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv').options(delimiter="\t", header='true', escape="\\").load('data/labeledTrainData.tsv')
df = df.withColumn("label", df["label"].cast(DoubleType()))
training, test = df.randomSplit([0.6, 0.4], seed=11)

bsParser = BeautifulSoupParser(inputCol="review", outputCol="parsed")
tokenizer = Tokenizer(inputCol=bsParser.getOutputCol(), outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.01)
pipeline = Pipeline(stages=[bsParser, tokenizer, hashingTF, lr])

model = pipeline.fit(training)

prediction = model.transform(test)

predictionAndLabels = prediction.map(lambda s: (s.prediction, s.label))

metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)

# Area under ROC curve
print("Area under ROC = %s" % metrics.areaUnderROC)
