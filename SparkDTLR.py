from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import col
import sys, time

class Reformatter(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, categorical_columns, encoded_columns, all_columns):
        super(Reformatter, self).__init__()
        self.categorical_columns = categorical_columns
        self.encoded_columns = encoded_columns
        self.all_columns = all_columns

    # Remove unwanted columns
    def _transform(self, df):
        for c in self.categorical_columns:
            if (c in self.encoded_columns):
                df = df.drop(c)
                df = df.drop("{}_indexed".format(c))
                df = df.withColumnRenamed("{}_encoded".format(c), c)
            else:
                df = df.drop(c)
                df = df.withColumnRenamed("{}_indexed".format(c), c)

        # Sort columns back into their original order
        df = df.select(*(self.all_columns))

        # Cast to int to prevent floating point values causing extra classes (edge case)
        df = df.withColumn("label", col("label").cast("int"))
        
        return df

def full_loop(train_data, test_data, pipeline):
    # Train, Test
    start_time = time.time()
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    end_time = time.time()

    # Evaluators
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    roc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")

    # Evaluate
    test_acc = acc_evaluator.evaluate(predictions)
    test_roc = roc_evaluator.evaluate(predictions)

    results = {"train_acc": model.stages[-1].summary.accuracy,
               "test_acc": test_acc,
               "train_roc": model.stages[-1].summary.areaUnderROC,
               "test_roc": test_roc,
               "runtime": end_time - start_time,
               "model_coefficients": model.stages[-1].coefficients,
               "model_intercept": model.stages[-1].intercept}
    
    return results

def print_and_append_results(heading, results, contents):
    print(heading)
    contents += "{}\n".format(heading)
    for key, value in results.items():
        str = "{}: {}".format(key, value)
        print(str)
        contents += "{}\n".format(str)
    print()
    contents += "\n"
    return contents

def main():
    if len(sys.argv) != 5:
        print("Usage: SparkDTLR.py <data_file> <schema_file> <output_dir> <seed>")
        sys.exit(-1)

    data_path = sys.argv[1]
    schema_path = sys.argv[2]
    output_path = sys.argv[3]
    seed = sys.argv[4]

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

    # Create splits
    train_data, test_data = data.randomSplit([0.7, 0.3], seed=seed)

    # Parse schema
    lines = sc.textFile(schema_path)
    lines = lines.collect()
    categorical_columns = ["_c{}".format(i) for i in range(len(lines)) if lines[i].endswith("symbolic.")] + ["label"]
    
    # Explicitly define columns to be one-hot encoded
    encoded_columns = [c for c in categorical_columns if ((train_data.select(c).distinct().count() > 1) and not (c == "label"))]

    # Logistic regression pipeline
    lr_indexers = [StringIndexer(inputCol=c, outputCol="{}_indexed".format(c), handleInvalid="keep") for c in categorical_columns]
    lr_encoders = [OneHotEncoder(inputCol="{}_indexed".format(c), outputCol="{}_encoded".format(c), handleInvalid="keep") for c in encoded_columns]
    lr_reformatter = Reformatter(categorical_columns, encoded_columns, all_columns)
    lr_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    lr_regression = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5, labelCol="label", featuresCol="features", family="binomial")
    lr_stages = lr_indexers + lr_encoders + [lr_reformatter, lr_assembler, lr_regression]
    lr_pipeline = Pipeline(stages=lr_stages)

    # Run loop and store results in dictionaries
    lr_results = full_loop(train_data, test_data, lr_pipeline)

    # Write to file
    contents = ""
    contents = print_and_append_results("LOGISTIC REGRESSION", lr_results, contents)
    #rdd = sc.parallelize([contents])
    #rdd.coalesce(1).saveAsTextFile(output_path)

if __name__ == "__main__":
    main()