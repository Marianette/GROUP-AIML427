from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, PCA
from pyspark.sql.functions import col
import sys, time

# Preprocessing rename cols, remove nulls
def preprocess(df):
    df = df.drop("_c24")
    df = df.dropna()
    feature_columns = ["c{}".format(i) for i in range(len(df.columns)-1)]
    all_columns = feature_columns + ["label"]
    return df.toDF(*all_columns)

# Transformer to rename encoded/indexed columns to match original names
class Reformatter(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, indexed_columns, encoded_columns, all_columns):
        super(Reformatter, self).__init__()
        self.indexed_columns = indexed_columns
        self.encoded_columns = encoded_columns
        self.all_columns = all_columns

    def _transform(self, df):
        for c in self.indexed_columns:
            if c in self.encoded_columns:
                df = df.drop(c)
                df = df.drop(f"{c}_indexed")
                df = df.withColumnRenamed(f"{c}_encoded", c)
            else:
                df = df.drop(c)
                df = df.withColumnRenamed(f"{c}_indexed", c)
        df = df.select(*self.all_columns)
        df = df.withColumn("label", col("label").cast("int"))
        return df

# Full pipeline: train, predict, evaluate
def full_loop(train_data, test_data, pipeline):
    start_time = time.time()
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    end_time = time.time()

    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    results = {
        "test_accuracy": acc_eval.evaluate(predictions),
        "test_roc_auc": roc_eval.evaluate(predictions),
        "runtime_sec": end_time - start_time,
        "tree_depth": model.stages[-1].depth,
        "num_nodes": model.stages[-1].numNodes
    }
    return results

# Print to stdout and append results to content string
def print_and_append_results(heading, results, contents):
    print(heading)
    contents += f"{heading}\n"
    for key, value in results.items():
        line = f"{key}: {value}"
        print(line)
        contents += f"{line}\n"
    contents += "\n"
    return contents

# Main method for invoking from CLI
def main():
    if len(sys.argv) != 4:
        print("Usage: SparkDecisionTree.py <train_file> <test_file> <output_dir>")
        sys.exit(-1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    spark = SparkSession.builder.appName("SparkDecisionTree").getOrCreate()
    sc = spark.sparkContext

    train_data = spark.read.options(delimeter=",", inferSchema=True).csv(train_path)
    test_data = spark.read.options(delimeter=",", inferSchema=True).csv(test_path)

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    all_columns = train_data.columns
    feature_columns = all_columns[:-1]

    indexed_columns = [c for c, dtype in train_data.dtypes if dtype == "string"]
    encoded_columns = [c for c in indexed_columns if train_data.select(c).distinct().count() > 1 and c != "label"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep") for c in indexed_columns]
    encoders = [OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_encoded", handleInvalid="keep") for c in encoded_columns]
    reformatter = Reformatter(indexed_columns, encoded_columns, all_columns)
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [reformatter, assembler, tree])

    results = full_loop(train_data, test_data, pipeline)

    contents = ""
    contents = print_and_append_results("DECISION TREE RESULTS", results, contents)
    sc.parallelize([contents]).coalesce(1).saveAsTextFile(output_path)

if __name__ == "__main__":
    main()
