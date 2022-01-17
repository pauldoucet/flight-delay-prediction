import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

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


    val df = spark.read.option("header", "true").csv("datasets/2009.csv")

    val columns = df.columns

    df.show()
    //val first = df.first()

    //println(first)
  }
}