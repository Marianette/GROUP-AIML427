# README 
SparkDTLR - Logistic Regression and Decision Tree Classifier on KDD Dataset

## Authors

Maria DaRocha (300399718), William Huang (300653623)

## Overview

This project evaluates two classifiers, Logistic Regression (LR) and Decision Tree (DT), using Apache Spark's MLlib. The classifiers are trained and tested on the KDD dataset for binary classification. The evaluation is repeated 10 times with different random seeds to assess performance consistency.

## Dependencies

* Python 3.6+
* PySpark 3.0+
* HDFS access
* Apache Spark runtime

## Files

* `SparkDTLR.py`: Main script that runs both classifiers across 10 seeds and outputs evaluation metrics.
* `kdd.data`: Input dataset file.
* `kdd.schema`: Schema file describing attribute types.

## Execution Instructions

### Step 1: Ensure Data is in HDFS

```bash
hdfs dfs -put kdd.data /user/yourusername/
hdfs dfs -put kdd.schema /user/yourusername/
```

### Step 2: Submit the Spark Job

```bash
spark-submit SparkDTLR.py kdd.data kdd.schema output_dir 296
```

Where:

* `kdd.data` is your dataset path
* `kdd.schema` is the schema path
* `output_dir` is where result files will be saved
* `296` is the base random seed

### Step 3: Retrieve Output

```bash
hdfs dfs -get output_dir/part-00000 result.txt
```

## Outputs

The script outputs the following per model and seed:

* Train/Test Accuracy
* Train/Test ROC AUC
* Train/Test Precision
* Train/Test Recall
* Train/Test F1 Score
* Runtime (in seconds)
* LR: Coefficients & Intercept
* DT: Tree Depth & Number of Nodes

Additionally, a summary section computes min, max, average, and standard deviation across all 10 seeds for each metric.

## Notes

* All categorical variables are indexed and encoded using `StringIndexer` and `OneHotEncoder`.
* A custom transformer (`Reformatter`) restores schema order and cleans up pipeline artifacts.
* Pipelines are reused for consistent processing between LR and DT.

## Acknowledgements

This work was conducted as part of the AIML427 course requirements. Dataset was pre-provided in the assignment brief and is available at the online UCI ML Library.
