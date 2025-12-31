ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version      := "0.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "spark_jobs",

    // Spark libs: marked "Provided" so they are not included in the JAR
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core"  % "3.5.1" % Provided,
      "org.apache.spark" %% "spark-sql"   % "3.5.1" % Provided,
      "org.apache.spark" %% "spark-mllib" % "3.5.1" % Provided,

      // XGBoost Spark
      "ml.dmlc" %% "xgboost4j-spark" % "1.7.6"
    ),

    // This line specifically skips running tests during the assembly process
    assembly / test := {},

    // REQUIRED: Handle merge conflicts for the Uber JAR
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x => MergeStrategy.first
    }
  )