import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

object TrainFraudXGB {

  case class Args(
    input: String = "s3a://datalake/gold/fraud_features",
    output: String = "s3a://datalake/models/fraud_xgb",
    numWorkers: Int = 2
  )

  def parseArgs(raw: Array[String]): Args = {
    def next(i: Int) = if (i + 1 < raw.length) raw(i + 1) else ""
    var a = Args()
    var i = 0
    while (i < raw.length) {
      raw(i) match {
        case "--input"      => a = a.copy(input = next(i)); i += 1
        case "--output"     => a = a.copy(output = next(i)); i += 1
        case "--numWorkers" => a = a.copy(numWorkers = next(i).toInt); i += 1
        case _ =>
      }
      i += 1
    }
    a
  }

  def main(args: Array[String]): Unit = {
    val conf = parseArgs(args)
    
    println(s"📊 Config: input=${conf.input}, output=${conf.output}, numWorkers=${conf.numWorkers}")

    val spark = SparkSession.builder()
      .appName("TrainFraudXGB")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()

    import spark.implicits._

    try {
      // ⚠️ FIX #1: Đọc data với cache để tránh đọc lại nhiều lần
      println("📖 Reading training data...")
      val df0 = spark.read.parquet(conf.input)
      
      println(s"📋 Schema:")
      df0.printSchema()
      
      // ⚠️ FIX #2: Validate columns
      val requiredCols = Set("label", "features", "event_date")
      val actualCols = df0.columns.toSet
      if (!requiredCols.subsetOf(actualCols)) {
        throw new IllegalArgumentException(
          s"Missing columns! Required: $requiredCols, Found: $actualCols"
        )
      }

      // ⚠️ FIX #3: Filter và cache ngay
      val df = df0
        .select(
          col("features"),
          col("label").cast("double").as("label"),
          col("event_date")
        )
        .filter(col("features").isNotNull && col("label").isNotNull)
        .cache()  // Cache để tránh đọc lại từ S3

      val n = df.count()
      println(s"✅ Loaded rows = $n")

      if (n == 0) {
        throw new IllegalStateException("No data found after filtering!")
      }

      // ⚠️ FIX #4: Kiểm tra class balance
      val labelCounts = df.groupBy("label").count().orderBy("label").collect()
      println("📊 Label distribution:")
      labelCounts.foreach { row =>
        val label = row.getDouble(0)
        val count = row.getLong(1)
        val pct = count * 100.0 / n
        println(f"  Label $label%.0f: $count%,d ($pct%.2f%%)")
      }

      // Split strategy
      val dates = df.select("event_date")
        .distinct()
        .orderBy(col("event_date").desc)
        .limit(3)
        .collect()
        .map(_.get(0).toString)
      
      val distinctDates = df.select("event_date").distinct().count()
      println(s"📅 Distinct dates: $distinctDates")
      println(s"📅 Recent dates: ${dates.mkString(", ")}")

      val (trainDf, testDf) = if (distinctDates >= 2) {
        val maxDate = df.select(max(col("event_date"))).as[String].head()
        println(s"🎯 Using temporal split: test date = $maxDate")
        
        val train = df.filter(col("event_date") =!= lit(maxDate)).cache()
        val test = df.filter(col("event_date") === lit(maxDate)).cache()
        
        val trainCount = train.count()
        val testCount = test.count()
        println(s"📊 Train: $trainCount, Test: $testCount")
        
        (train, test)
      } else {
        println("⚠️  Only 1 event_date, using random split")
        val Array(tr, te) = df.randomSplit(Array(0.9, 0.1), seed = 42)
        val trainCount = tr.count()
        val testCount = te.count()
        println(s"📊 Train: $trainCount, Test: $testCount")
        (tr, te)
      }

      // Unpersist original df
      df.unpersist()

      // ⚠️ FIX #5: XGBoost params tối ưu cho memory
      println("🏋️  Training XGBoost model...")
      val xgbParams = Map(
        "max_depth" -> 5,
        "eta" -> 0.1,
        "min_child_weight" -> 5.0,
        "subsample" -> 0.8,
        "colsample_bytree" -> 0.8,
        "objective" -> "binary:logistic",
        "eval_metric" -> "auc",
        "tree_method" -> "hist",
        "max_bin" -> 256,
        "nthread" -> 1,

        // ✅ FIX lỗi bạn đang gặp:
        "missing" -> Float.NaN,
        "allow_non_zero_for_missing" -> true
      )
      
      
      val xgb = new XGBoostClassifier(xgbParams)
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setNumWorkers(conf.numWorkers)
        .setNumRound(50)


      val model = xgb.fit(trainDf)
      println("✅ Model trained successfully!")

      // ⚠️ FIX #6: Evaluate với repartition để tránh OOM
      println("📈 Evaluating model...")
      val scored = model.transform(testDf).repartition(2)
      
      val eval = new BinaryClassificationEvaluator()
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")

      val auc = eval.evaluate(scored)
      println(f"✅ AUC = $auc%.6f")

      // Confusion matrix
      val predictions = scored
        .select(col("label"), col("prediction"))
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .collect()

      println("📊 Confusion Matrix:")
      predictions.foreach { row =>
        val label = row.getDouble(0)
        val pred = row.getDouble(1)
        val count = row.getLong(2)
        println(f"  Label $label%.0f, Pred $pred%.0f: $count%,d")
      }

      // ⚠️ FIX #7: Save model với proper error handling
      println(s"💾 Saving model to ${conf.output}...")
      try {
        model.write.overwrite().save(conf.output)
        println(s"✅ Model saved successfully!")
      } catch {
        case e: Exception =>
          println(s"❌ Failed to save model: ${e.getMessage}")
          e.printStackTrace()
          throw e
      }

      // Cleanup
      trainDf.unpersist()
      testDf.unpersist()

    } catch {
      case e: Exception =>
        println(s"❌ Training failed: ${e.getMessage}")
        e.printStackTrace()
        throw e
    } finally {
      spark.stop()
      println("🏁 Spark session stopped")
    }
  }
}