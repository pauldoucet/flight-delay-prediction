import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.classification.SVMWithSGD
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


    var df = spark.read.option("header", "true").csv("datasets/2009.csv")

    df = df.select("DEP_TIME", "DEP_DELAY", "TAXI_OUT")

    val rdd = df.rdd

    val splits = rdd.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numIterations = 100

    df.show()
  }
}