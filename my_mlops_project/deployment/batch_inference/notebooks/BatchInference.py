# Databricks notebook source
# Install the databricks-feature-engineering package
%pip install databricks-feature-engineering lightgbm

# Restart Python to ensure the package is loaded correctly
dbutils.library.restartPython()

# COMMAND ----------

##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# It is configured and can be executed as the batch_inference_job in the batch_inference_job workflow defined under
# ``my_mlops_project/resources/batch-inference-workflow-resource.yml``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
#  * model_name (required) - The name of the model to be used in batch inference.
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# A Hive-registered Delta table containing the input features.
dbutils.widgets.text("input_table_name", "", label="Input Table Name")
# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "", label="Output Table Name")
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.my_mlops_project.my_mlops_project-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

#%pip install -r ../../../requirements.txt

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1,Define input and output variables

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
model_name = dbutils.widgets.get("model_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

# DBTITLE 1,Prepare Batch Inference Input Table
from pyspark.sql.functions import to_timestamp, lit
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta, timezone




def rounded_unix_timestamp(dt, num_minutes=15):
  """
  Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
  """
  nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
  delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
  return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())


df = spark.table("delta.`dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled`")
df.withColumn(
  "rounded_pickup_datetime",
  to_timestamp(rounded_unix_timestamp_udf(df["tpep_pickup_datetime"], lit(15))),
).withColumn(
  "rounded_dropoff_datetime",
  to_timestamp(rounded_unix_timestamp_udf(df["tpep_dropoff_datetime"], lit(30))),
).drop(
  "tpep_pickup_datetime"
).drop(
  "tpep_dropoff_datetime"
).drop(
  "fare_amount"
).write.mode(
  "overwrite"
).saveAsTable(
  name=input_table_name
)


# COMMAND ----------

from mlflow import MlflowClient

# Get model version from alias
client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

display(model_version)

# COMMAND ----------

# DBTITLE 1,Load model and run inference
from predict import predict_batch

predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
dbutils.notebook.exit(output_table_name)
