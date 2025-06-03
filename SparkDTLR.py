from pyspark.sql import SparkSession
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Normalizer, PCA
from pyspark.sql.functions import col
import sys, time

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

    # Parse schema
    lines = sc.textFile(schema_path)
    lines = lines.collect()
    categorical_columns = [i for i in range(len(lines)) if lines[i].endswith("symbolic.")]
    print(categorical_columns)

if __name__ == "__main__":
    main()