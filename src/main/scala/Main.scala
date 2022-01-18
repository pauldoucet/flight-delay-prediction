import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, TrainValidationSplit}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{SQLContext, SparkSession}

/**
 * @author Paul Doucet (316442)
 */
object Main {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/winutils/")

    val spark = SparkSession
      .builder()
      .appName("flight-delay-prediction")
      .config("spark.master", "local")
      .getOrCreate()


    val feature_cols = Array("DEP_TIME", "DEP_DELAY", "TAXI_OUT")

    val df = spark.read.option("header", "true")
      .csv("datasets/2009.csv")
      .withColumn("DEP_TIME", col("DEP_TIME").cast("double"))
      .withColumn("DEP_DELAY", col("DEP_DELAY").cast("double"))
      .withColumn("TAXI_OUT", col("TAXI_OUT").cast("double"))
      .withColumn("ARR_DELAY", col("ARR_DELAY").cast("double"))
      .select("DEP_TIME", "DEP_DELAY", "TAXI_OUT", "ARR_DELAY")
      .na.drop()

    //df.dtypes.foreach(s => print(s"{${s._1}, ${s._2}}"))

    //df.show()

    val assembler = new VectorAssembler()
      .setHandleInvalid("skip")
      .setInputCols(feature_cols).setOutputCol("FEATURES")
    //df = assembler.transform(df)

    /*val splits = df.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)*/

    val numIterations = 100

    val lr = new LinearRegression()
      .setFeaturesCol("FEATURES")
      .setLabelCol("ARR_DELAY")
      .setMaxIter(numIterations)

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())



    val predicted = model.transform(test)

    predicted.show()
  }
}