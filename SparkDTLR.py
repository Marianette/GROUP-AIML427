from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import col
import sys, time
import numpy as np

# Custom transformer to rename encoded/indexed columns to match original names
class Reformatter(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, categorical_columns, encoded_columns, all_columns):
        super(Reformatter, self).__init__()
        self.categorical_columns = categorical_columns
        self.encoded_columns = encoded_columns
        self.all_columns = all_columns

    # Remove unwanted columns
    def _transform(self, df):
        for c in self.categorical_columns:
            if c in self.encoded_columns:
                df = df.drop(c)
                df = df.drop(f"{c}_indexed")
                df = df.withColumnRenamed(f"{c}_encoded", c)
            else:
                df = df.drop(c)
                df = df.withColumnRenamed(f"{c}_indexed", c)

        # Sort columns back into their original order
        df = df.select(*self.all_columns)
        # Cast to int to prevent floating point values causing extra classes (edge case)
        df = df.withColumn("label", col("label").cast("int"))
        return df

# Executes pipeline and returns evaluation metrics and metadata
def full_loop(train_data, test_data, pipeline):
    # Train, Test
    start_time = time.time()
    model = pipeline.fit(train_data)
    predictions_test = model.transform(test_data)
    predictions_train = model.transform(train_data)
    end_time = time.time()

    # Evaluators
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    # Evaluate
    results = {
        "train_accuracy": acc_eval.evaluate(predictions_train),
        "test_accuracy": acc_eval.evaluate(predictions_test),
        "train_roc_auc": roc_eval.evaluate(predictions_train),
        "test_roc_auc": roc_eval.evaluate(predictions_test),
        "train_precision": prec_eval.evaluate(predictions_train),
        "test_precision": prec_eval.evaluate(predictions_test),
        "train_recall": rec_eval.evaluate(predictions_train),
        "test_recall": rec_eval.evaluate(predictions_test),
        "train_f1": f1_eval.evaluate(predictions_train),
        "test_f1": f1_eval.evaluate(predictions_test),
        "runtime_sec": end_time - start_time,
    }

    if hasattr(model.stages[-1], "coefficients"):
        results["model_coefficients"] = model.stages[-1].coefficients
        results["model_intercept"] = model.stages[-1].intercept
    if hasattr(model.stages[-1], "depth"):
        results["tree_depth"] = model.stages[-1].depth
        results["num_nodes"] = model.stages[-1].numNodes

    return results

# Print to stdout and also append results to content string
def print_and_append_results(heading, results, contents):
    print(heading)
    contents += f"{heading}\n"
    for key, value in results.items():
        line = f"{key}: {value}"
        print(line)
        contents += f"{line}\n"
    contents += "\n"
    return contents

# Main function

def main():
    if len(sys.argv) != 5:
        print("Usage: SparkDTLR.py <data_file> <schema_file> <output_dir> <base_seed>")
        sys.exit(-1)

    data_path = sys.argv[1]
    schema_path = sys.argv[2]
    output_path = sys.argv[3]
    base_seed = int(sys.argv[4])

    # Start spark session
    spark = SparkSession.builder.appName("SparkDTLR").getOrCreate()
    sc = spark.sparkContext

    # Read data
    data = spark.read.options(delimeter=",", inferSchema=True).csv(data_path)
    data = data.withColumnRenamed("_c41", "label")

    # Reference of columns that should exist in the dataframe
    all_columns = data.columns
    feature_columns = all_columns.copy()
    feature_columns.remove("label")

    # Parse schema
    lines = sc.textFile(schema_path).collect()
    categorical_columns = ["_c{}".format(i) for i, line in enumerate(lines) if line.endswith("symbolic.")] + ["label"]
    encoded_columns = [c for c in categorical_columns if data.select(c).distinct().count() > 1 and c != "label"]

    # Explicitly index and define columns to be one-hot encoded
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep") for c in categorical_columns]
    encoders = [OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_encoded", handleInvalid="keep") for c in encoded_columns]
    reformatter = Reformatter(categorical_columns, encoded_columns, all_columns)
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Logistic regression / Decision tree pipelines
    lr = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5, labelCol="label", featuresCol="features", family="binomial")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    lr_pipeline = Pipeline(stages=indexers + encoders + [reformatter, assembler, lr])
    dt_pipeline = Pipeline(stages=indexers + encoders + [reformatter, assembler, dt])

    metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
    all_results = {model: {f"train_{m}": [] for m in metrics} | {f"test_{m}": [] for m in metrics} | {"runtime_sec": []} for model in ["lr", "dt"]}

    contents = ""

    # Run loop and store results
    for i in range(10):
        seed = base_seed + i
        # Train/Test Split
        train_data, test_data = data.randomSplit([0.7, 0.3], seed=seed)

        lr_results = full_loop(train_data, test_data, lr_pipeline)
        dt_results = full_loop(train_data, test_data, dt_pipeline)

        for key in all_results["lr"]:
            all_results["lr"][key].append(lr_results[key])
            all_results["dt"][key].append(dt_results[key])

        contents = print_and_append_results(f"SEED {seed} - LOGISTIC REGRESSION", lr_results, contents)
        contents = print_and_append_results(f"SEED {seed} - DECISION TREE", dt_results, contents)

    for model in ["lr", "dt"]:
        contents += f"{model.upper()} SUMMARY\n"
        for key in all_results[model]:
            values = all_results[model][key]
            contents += f"{key} - min: {np.min(values)}\n"
            contents += f"{key} - max: {np.max(values)}\n"
            contents += f"{key} - avg: {np.mean(values)}\n"
            contents += f"{key} - std: {np.std(values)}\n"
        contents += "\n"

    sc.parallelize([contents]).coalesce(1).saveAsTextFile(output_path)

if __name__ == "__main__":
    main()

