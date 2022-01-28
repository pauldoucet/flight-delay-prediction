import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

/**
 * @author Paul Doucet (316442)
 */
object Main {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/winutils/")

    val spark = SparkSession
      .builder()
      .appName("flight-delay-prediction")
      .config("spark.master", "local[*]")
      .getOrCreate()


    val feature_cols = Array("DEP_TIME", "DEP_DELAY", "TAXI_OUT")

    val df = spark.read.option("header", "true")
      //.csv("s3://flight-delays-analytic/datasets/2009.csv")
      .csv("datasets/2009.csv")
      .withColumn("DEP_TIME", col("DEP_TIME").cast("double"))
      .withColumn("DEP_DELAY", col("DEP_DELAY").cast("double"))
      .withColumn("TAXI_OUT", col("TAXI_OUT").cast("double"))
      .withColumn("ARR_DELAY", col("ARR_DELAY").cast("double"))
      .select("DEP_TIME", "DEP_DELAY", "TAXI_OUT", "ARR_DELAY")
      .na.drop()

    val assembler = new VectorAssembler()
      .setHandleInvalid("skip")
      .setInputCols(feature_cols).setOutputCol("FEATURES")

    val numIterations = 100

    val lr = new LinearRegression()
      .setFeaturesCol("FEATURES")
      .setLabelCol("ARR_DELAY")
      .setMaxIter(numIterations)

    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    val splits = df.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val model = pipeline.fit(training)

    val predicted = model.transform(test)

    val r2 = new RegressionEvaluator()
      .setLabelCol("ARR_DELAY")
      .setPredictionCol("prediction")
      .setMetricName("r2")
      .evaluate(predicted)

    println(s"model got r^2 = $r2")

    System.in.read
    spark.stop // spark --> SparkSession
  }
}