from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datasources import Datasources
from pipelines import BaselinePipelineWrapper

sc = SparkContext("local", "Pipeline")
sqlContext = SQLContext(sc)

dt = Datasources(sc)
pipeline_wrapper = BaselinePipelineWrapper()

original_training_set = dt.get_original_training_set()
original_test_set = dt.get_original_test_set()

param_grid = ParamGridBuilder().addGrid(pipeline_wrapper.hashing_tf.numFeatures, [pow(2, 20)]).addGrid(pipeline_wrapper.lr.regParam, [0.1, 0.01]).build()
cv = CrossValidator().setEstimator(pipeline_wrapper.pipeline).setEvaluator(BinaryClassificationEvaluator()).setEstimatorParamMaps(param_grid).setNumFolds(3)

model = cv.fit(original_training_set)

prediction = model.transform(original_test_set)

id_label = prediction.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
id_label.saveAsTextFile("predictions.csv")
