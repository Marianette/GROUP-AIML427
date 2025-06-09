from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, PCA
from pyspark.sql.functions import col
import sys, time

def preprocess(df):
    # Clean up data
    df = df.drop("_c24")
    df = df.dropna()

    # Rename columns
    feature_columns = ["c{}".format(i) for i in range(len(df.columns)-1)]
    all_columns = feature_columns + ["label"]
    df = df.toDF(*all_columns)
    
    return df

class Reformatter(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, indexed_columns, encoded_columns, all_columns):
        super(Reformatter, self).__init__()
        self.indexed_columns = indexed_columns
        self.encoded_columns = encoded_columns
        self.all_columns = all_columns

    # Remove unwanted columns
    def _transform(self, df):
        for c in self.indexed_columns:
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
    if len(sys.argv) != 4:
        print("Usage: SparkLogisticRegression.py <train_file> <test_file> <output_dir>")
        sys.exit(-1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    # Start spark session
    spark = SparkSession.builder.appName("SparkLogisticRegression").getOrCreate()
    sc = spark.sparkContext

    # Read data
    train_data = spark.read.options(delimeter=", ", inferSchema=True).csv(train_path)
    test_data = spark.read.options(delimeter=", ", inferSchema=True).csv(test_path)

    # Preprocess data
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    # Reference of columns that should exist in the dataframe
    all_columns = train_data.columns
    feature_columns = all_columns.copy()
    feature_columns.remove("label")

    # Explicitly define column names for encoding categorical variables
    indexed_columns = []
    for c, dtype in train_data.dtypes:
        if (dtype == "string"):
            indexed_columns.append(c)
    encoded_columns = []
    for c in indexed_columns:
        if ((train_data.select(c).distinct().count() > 1) and not (c == "label")):
            encoded_columns.append(c)

    # Base pipeline
    baseline_indexers = [StringIndexer(inputCol=c, outputCol="{}_indexed".format(c), handleInvalid="keep") for c in indexed_columns]
    baseline_encoders = [OneHotEncoder(inputCol="{}_indexed".format(c), outputCol="{}_encoded".format(c), handleInvalid="keep") for c in encoded_columns]
    baseline_reformatter = Reformatter(indexed_columns, encoded_columns, all_columns)
    baseline_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    baseline_regression = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5, labelCol="label", featuresCol="features", family="binomial")
    baseline_stages = baseline_indexers + baseline_encoders + [baseline_reformatter, baseline_assembler, baseline_regression]
    baseline_pipeline = Pipeline(stages=baseline_stages)

    # Normalised pipeline
    normalised_indexers = [StringIndexer(inputCol=c, outputCol="{}_indexed".format(c), handleInvalid="keep") for c in indexed_columns]
    normalised_encoders = [OneHotEncoder(inputCol="{}_indexed".format(c), outputCol="{}_encoded".format(c), handleInvalid="keep") for c in encoded_columns]
    normalised_reformatter = Reformatter(indexed_columns, encoded_columns, all_columns)
    normalised_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    normaliser = Normalizer(inputCol="features", outputCol="normFeatures")
    normalised_regression = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5, labelCol="label", featuresCol="normFeatures", family="binomial")
    normalised_stages = normalised_indexers + normalised_encoders + [normalised_reformatter, normalised_assembler, normaliser, normalised_regression]
    normalised_pipeline = Pipeline(stages=normalised_stages)

    # PCA pipeline
    pca_indexers = [StringIndexer(inputCol=c, outputCol="{}_indexed".format(c), handleInvalid="keep") for c in indexed_columns]
    pca_encoders = [OneHotEncoder(inputCol="{}_indexed".format(c), outputCol="{}_encoded".format(c), handleInvalid="keep") for c in encoded_columns]
    pca_reformatter = Reformatter(indexed_columns, encoded_columns, all_columns)
    pca_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
    pca_regression = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.5, labelCol="label", featuresCol="pcaFeatures", family="binomial")
    pca_stages = pca_indexers + pca_encoders + [pca_reformatter, pca_assembler, pca, pca_regression]
    pca_pipeline = Pipeline(stages=pca_stages)

    # Run loop and store results in dictionaries
    baseline_results = full_loop(train_data, test_data, baseline_pipeline)
    normalised_results = full_loop(train_data, test_data, normalised_pipeline)
    pca_results = full_loop(train_data, test_data, pca_pipeline)

    # Write to file
    contents = ""
    contents = print_and_append_results("BASELINE", baseline_results, contents)
    contents = print_and_append_results("NORMALISED", normalised_results, contents)
    contents = print_and_append_results("PCA", pca_results, contents)
    rdd = sc.parallelize([contents])
    rdd.coalesce(1).saveAsTextFile(output_path)

if __name__ == "__main__":
    main()
