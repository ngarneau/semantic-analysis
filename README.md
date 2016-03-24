# Semantic analysis on Spark

To run the Spark application, assuming Spark binaries are in your path:

```
$ spark-submit --packages com.databricks:spark-csv_2.11:1.4.0 application.py
```

To test the pipelines and datasource files:

```
$ spark-submit --packages com.databricks:spark-csv_2.11:1.4.0 datasources.py
```

```
$ spark-submit --packages com.databricks:spark-csv_2.11:1.4.0 pipelines.py
```
