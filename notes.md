# Intro

This markdown file contains an overview of the code I have written in SparkDTLR.py. I'll try to break it down as much as possible and explain things that might not be so intuitive.

# Reading in the data, and manipulating dataframes

When you read in a csv file without headers, the resulting dataframe will automatically assign column names "_c0", "_c1", "_c2", etc.

A lot of the transformation operations do not allow us to overwrite existing input columns with transformed output columns, so as we're passing our dataframe through all of these operations, more and more columns are added, and keeping track of their names gets annoying. That's why I have included a custom transformer class, called the Reformatter (more on this later).

# Encoding categorical variables

We start by identifying the column names of the categorical variables, according to the schema.

All categorical variables must be converted into numerical form with the aid of the StringIndexer transformer.

I then choose to apply one-hot encoding for better classification performance. Not all categorical variables will be one-hot encoded - only the ones for which the training data observes more than 1 distinct category (and also excluding the label).

There is no batch operation for the StringIndexer and OneHotEncoder transformer on multiple columns at once, so we instead create batches of them that each operate on a single column, and chain them together using the pipeline.

Also, note that the encodings are generated from the training data, and when they are applied to the test data, they will end up encountering unseen values. This is a normal occurrence, and we are equipped to handle it by assigning the "handleInvalid" parameter of StringIndexer and OneHotEncoder to "keep".

# Reformatter

As mentioned, we end up adding columns to the dataframe with every transformation. This leaves us with added columns from StringIndexer and from OneHotEncoder. Reformatter takes as input the original names of the columns that have undergone these transformations, and then cleans up the dataframe and assigns those column names to their most pertinent newly added columns.

The "label" column should contain only binary values. But, a strange edge case that happened to me at one point was that the computer-precision floating-point values of certain entries ended up being recognised as extra classes, causing errors with "more than 2 classes being present". This is why I cast it to int.

# Using the pipeline

The function "full_loop" is where training, testing, and evaluation happens. It's as simple as fitting the pipeline to the training data and then using it to transform the test data.

We can get training evaluation metrics by pulling them out from the object corresponding to whichever algorithm we're evaluating (LogisticRegression, DecisionTreeClassifier). We access that object from the list of stages we assign to the pipeline. Then, the metrics are just attributes of the object.

For test evaluation metrics, we have to apply a separate evaluator class that is set to give us the metric we want. The ones I have currently are the ones I'm using for part 2. There is a bit of mismatch between which metrics are offered for which algorithms and evaluator classes, which is a bit annoying - we may not be able to use all of the ones you have suggested.

# Interpreting the model

The model coefficients are a sparse vector, which is a tuple with three entries. The first is a single number, indicating the number of model coefficients there are. The second is a list, indicating the indices of the coefficients that are nonzero. The third is a list, indicating the values of those nonzero coefficients.

# Writing results to file

The way I have set it up is that all of the data is extracted from results dictionaries for each run, and flattened into one text. If you want to observe how the file is written in your local space, you can easily implement a native python file writer. But this doesn't work on the Spark cluster, because it uses the HDFS, and ends up getting the spaces mixed up. We have to use Spark methods to write to text files, but obviously, those won't work in your local space, so I have commented them out for now.

# How to run the code

Call the following line in your terminal:

`python SparkDTLR.py kdd.data kdd.schema . <seed number>`