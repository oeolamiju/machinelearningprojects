# Databricks notebook source
dbutils.fs.ls('/FileStore/tables/FaultDataset.csv')

# COMMAND ----------

#importing mlflow to autolog machine learning runs

import mlflow

mlflow.pyspark.ml.autolog()

# COMMAND ----------

#reading data from the FaultDataset.csv file into a new spark dataframe

fault_DF = spark.read.csv("dbfs:/FileStore/tables/FaultDataset.csv", header = "true", inferSchema = "true")

# COMMAND ----------

#show the content of the created dataframe
fault_DF.display(3)

# COMMAND ----------


#import the count and desc funtions from pyspark.sql
from pyspark.sql.functions import count, desc

#count the faults (1) and no faults (0) in the 'fault_detected' column and group by the same column, then arrange in descending order.
fault_DF.groupBy('fault_detected').agg(count('*').alias('count')).orderBy(desc('count')).show()

# COMMAND ----------

#creating a temporary view from fault_DF dataframe
fault_DF.createOrReplaceTempView ("fault_sql")

# COMMAND ----------

# MAGIC %sql
# MAGIC --counting the total number of records
# MAGIC SELECT COUNT(*) AS Sets_of_Readings FROM fault_sql

# COMMAND ----------

# MAGIC %sql
# MAGIC Select fault_detected, Count(fault_detected) From fault_sql
# MAGIC Group By fault_detected

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from fault_sql

# COMMAND ----------

#for preprocessing the data, we import the RFormula Transformer to transform the data into a vector format for MLlib to be able to train.

from pyspark.ml.feature import RFormula

preprocess = RFormula(formula="fault_detected ~ .")

fault_DF = preprocess.fit(fault_DF).transform(fault_DF)

fault_DF.show(5)

# COMMAND ----------

#splitting the data randomly into 70% and 30% for the training and test dataframes respectively

(trainingDF, testDF) = fault_DF.randomSplit([0.7, 0.3], seed=100)

# COMMAND ----------


#the DecisionTreeClassifier estimator is imported from MLlib to serve as the classification model. The algorithm uses the training dataframe to construct a decision tree

from pyspark.ml.classification import DecisionTreeClassifier

#instantiating an instance of the estimator with the label and the features columns.
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features")

#training the model

model = dt.fit(trainingDF)

# COMMAND ----------

#making predictions on the test dataframe

predictions = model.transform(testDF)

predictions.select('rawPrediction', 'probability', 'prediction', 'fault_detected', 'features').show(5)

# COMMAND ----------

#import the count and desc funtions from pyspark.sql
from pyspark.sql.functions import count, desc

#count each prediction in the 'prediction' column of the predictions dataframe above and group by the same column, then arrange in descending order.
predictions.groupBy('prediction').agg(count('*').alias('count')).orderBy(desc('count')).show()

# COMMAND ----------

#using the evaluator to measure the accuracy of predictions on the test data

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)

print("Accuracy = %g " % (accuracy))

# COMMAND ----------

#importing and instantiating the ParamGridBuilder to take in hyperparameters for grid search
from pyspark.ml.tuning import ParamGridBuilder

#creating a parameter grid for hyperparameter tuning.
parameters = ParamGridBuilder()\
.addGrid(dt.impurity,["gini", "entropy"])\
.addGrid(dt.maxDepth, [3, 5, 7])\
.addGrid(dt.maxBins, [16, 32, 64])\
.build()

# COMMAND ----------

#importing and instantiating the TrainValidationSplit to split the dataset into training and validation datasets.

from pyspark.ml.tuning import TrainValidationSplit
#Defining TrainValidationSplit
tvs = TrainValidationSplit()\
.setSeed(100)\
.setTrainRatio(0.75)\
.setEstimatorParamMaps(parameters)\
.setEstimator(dt)\
.setEvaluator(evaluator)

# COMMAND ----------

#training the model using grid search
gridsearchModel = tvs.fit(trainingDF)

# COMMAND ----------

#selecting the best model and identifying the parameters

bestModel = gridsearchModel.bestModel

print("Parameters for the best model:")
print("MaxDepth Parameter: %g" %bestModel.getMaxDepth())
print("Impurity Parameter: %s" %bestModel.getImpurity())
print("MaxBins Parameter: %g" %bestModel.getMaxBins())

# COMMAND ----------

#Using the identified best model to make predictions on the hold out test set

evaluator.evaluate(bestModel.transform(testDF))

# COMMAND ----------

#using MLflow tracking to load a trained model in a given MLflow run and further use it to make predictions.
import mlflow
logged_model = 'runs:/34215fdf07e64a5686e474329ffc631a/best_model'
# Load model
loaded_model = mlflow.spark.load_model(logged_model)
# Perform inference via model.transform()
loaded_predictions = loaded_model.transform(testDF)
loaded_predictions.select("fault_detected", "rawPrediction", "probability", "prediction").show(10)
