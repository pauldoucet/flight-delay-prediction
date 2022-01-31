import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
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


    // Set of input features column names
    val feature_cols = Array("DEP_TIME", "DEP_DELAY", "TAXI_OUT")

    // Load csv file as a spark Dataframe
    // we convert every column from string to double
    // we only select features useful for training our model (i.e. "DEP_TIME", "DEP_DELAY", "TAXI_OUT")
    // we also select the output column ("ARR_DELAY")
    // Finally we drop any column containing null elements
    val df = spark.read.option("header", "true")
      //.csv("s3://flight-delays-analytic/datasets/2009.csv")
      .csv("datasets/2009.csv")
      .withColumn("DEP_TIME", col("DEP_TIME").cast("double"))
      .withColumn("DEP_DELAY", col("DEP_DELAY").cast("double"))
      .withColumn("TAXI_OUT", col("TAXI_OUT").cast("double"))
      .withColumn("ARR_DELAY", col("ARR_DELAY").cast("double"))
      .select("DEP_TIME", "DEP_DELAY", "TAXI_OUT", "ARR_DELAY")
      .na.drop()

    // We create a vector assembler to gather all out input columns in a single "FEATURES" column
    val assembler = new VectorAssembler()
      .setHandleInvalid("skip")
      .setInputCols(feature_cols).setOutputCol("FEATURES")

    val numIterations = 100

    // We instantiate the linear regression model
    val lr = new LinearRegression()
      .setFeaturesCol("FEATURES")
      .setLabelCol("ARR_DELAY")
      .setMaxIter(numIterations)

    // pipeline containing the features column assembling phase, and our linear regression model
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // empty parameter grid
    val paramGrid = new ParamGridBuilder()
      .build()

    // we train our model with a trainValidationSplit which will optimize the hyper-parameters
    // using a validation set
    val trainValidator = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("ARR_DELAY"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model = trainValidator.fit(df)
  }
}