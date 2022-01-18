name := "flight-delay-prediction"

version := "0.1"

scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0"
)