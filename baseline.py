from pyspark import SparkContext
from pyspark.sql import SQLContext
from datasources import Datasources
from pipelines import BaselinePipelineEngine

sc = SparkContext("local", "Pipeline", pyFiles=["baseline.py", "datasources.py", "transformers.py", "pipelines.py"])
sqlContext = SQLContext(sc)

dt = Datasources(sc)
pipeline_engine = BaselinePipelineEngine()

original_training_set = dt.get_original_training_set()
original_training_set = original_training_set.limit(10)
original_test_set = dt.get_original_test_set()
original_test_set = original_test_set.limit(10)

model = pipeline_engine.fit(original_training_set)
prediction = model.transform(original_test_set)

id_label = prediction.map(lambda s: '"' + s.id + '",' + str(int(s.prediction)))
id_label.saveAsTextFile("predictions.csv")
