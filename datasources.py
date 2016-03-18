import unittest
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import DoubleType
"""
Class that gathers different datasources
"""


class Datasources:
    def __init__(self, sc):
        self.sc = sc
        self.sql_context = SQLContext(sc)

    def _get_origin_dataset_loading_parameters(self):
        return self.sql_context.read.format('com.databricks.spark.csv').options(delimiter="\t", header='true', escape="\\")

    def get_original_training_set(self):
        dt = self._get_origin_dataset_loading_parameters().load('data/labeledTrainData.tsv')
        return dt.withColumn("label", dt["label"].cast(DoubleType()))

    def get_original_test_set(self):
        return self._get_origin_dataset_loading_parameters().load('data/testData.tsv')

    def save_test_predictions(self, predictions):
        predictions.select('id', 'prediction').write \
            .format('com.databricks.spark.csv') \
            .save('predictions.csv')


    def get_twitter_dataset(self):
        # Todo
        # May be interesting to have multiple datasets
        pass


class DatasourcesTest(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContext("local", "test_app")
        self.dt = Datasources(self.sc)

    def tearDown(self):
        self.sc.stop()

    def test_sql_context_init(self):
        self.assertIsInstance(self.dt.sql_context, SQLContext)

    def test_get_original_dataset(self):
        original_dataset = self.dt.get_original_training_set()
        self.assertIn('id', original_dataset.columns)
        self.assertIn('label', original_dataset.columns)
        self.assertIn('review', original_dataset.columns)
        self.assertEqual(25000, original_dataset.count())

    def test_original_dataset_label_is_float(self):
        original_dataset = self.dt.get_original_training_set()
        row = original_dataset.first()
        self.assertIsInstance(row.label, float)

    def test_get_original_dataset(self):
        test_dataset = self.dt.get_original_test_set()
        self.assertIn('id', test_dataset.columns)
        self.assertIn('review', test_dataset.columns)
        self.assertEqual(25000, test_dataset.count())


if __name__ == "__main__":
    unittest.main()
