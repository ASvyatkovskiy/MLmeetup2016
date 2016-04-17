from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import RegexTokenizer, Tokenizer
from pyspark.ml.feature import NGram
from pyspark.ml.feature import StopWordsRemover

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import StringIndexer, VectorIndexer

from pyspark.ml import Pipeline
from pyspark.sql import Row

import sys
import time

#Paths to train data and labels
PATH_TO_JSON = "file:///home/alexeys/MLmeetup2016/preprocess/html.avro"
PATH_TO_TRAIN_LABELS = "file:///home/alexeys/MLmeetup2016/preprocess/labels.avro"

def main(argv):
    start = time.time()

    #INGEST DATA INTO DATA FRAME OR TEMP. TABLE
    print "Ingest data..."
    sc = SparkContext(appName="KaggleDato")
    sqlContext = SQLContext(sc)

    train_label_df = sqlContext.read.format('com.databricks.spark.avro').load(PATH_TO_TRAIN_LABELS)
    input_df = sqlContext.read.format('com.databricks.spark.avro').load(PATH_TO_JSON)
    #input_df.printSchema()
    #train_label_df.printSchema()
    #input_df.show()

    #Make DF with labels    
    train_wlabels_df = input_df.join(train_label_df,"id")
    #train_wlabels_df.printSchema()

    #train CV split, stratified sampling
    #1 is under represented class
    fractions = {1.0:1.0, 0.0:1.0}
    stratified = train_wlabels_df.sampleBy("label", fractions, 36L)
    stratified = stratified.repartition(200)
    train, cv = stratified.randomSplit([0.7, 0.3])

    print "Prepare text features..."
    # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
    #tokenizer = Tokenizer(inputCol="text", outputCol="words")
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    #tokenized_df = tokenizer.transform(train_wlabels_df)
    #tokenized_df.show()

    #remove stopwords 
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    #filtered_df = remover.transform(tokenized_df)
    #filtered_df.printSchema()
    #filtered_df.show()

    #try ngrams instead
    #ngram = NGram(n=2, inputCol="filtered", outputCol="filtered")
    #ngram_df = ngram.transform(tokenized_df_copy)

    #Hashing
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
    #featurized_df = hashingTF.transform(filtered_df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    #idfModel = idf.fit(featurized_df)
    #rescaled_df = idfModel.transform(featurized_df)
    #rescaled_df.printSchema()

    #Trying various classifiers here
    #create a pipeline
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # Train a RandomForest model.

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    #labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    #featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2)

    #rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",numTrees=10,impurity="gini",maxDepth=4,maxBins=32)
    #pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, labelIndexer, featureIndexer, rf])

    #Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    #is areaUnderROC.
    #metricName options are: areaUnderROC|areaUnderPR)
    metricName = "areaUnderPR"
    ev = BinaryClassificationEvaluator(metricName=metricName)
    #Alternative: user multiclass classification evaluator
    #metricName options are f1, precision, recall
    #ev = MulticlassClassificationEvaluator(metricName="f1")

    # Fit the pipeline to training documents.
    model = pipeline.fit(train)

    print "Evaluate model on test instances and compute test error..."
    prediction = model.transform(cv)
    prediction.select("label", "text", "probability", "prediction").show(100)

    result = ev.evaluate(prediction)
    print metricName,": ", result

    cvErr = prediction.filter(prediction.label == prediction.prediction).count() / float(cv.count())
    print 'CV Error = ' + str(cvErr)

if __name__ == "__main__":
   main(sys.argv)
