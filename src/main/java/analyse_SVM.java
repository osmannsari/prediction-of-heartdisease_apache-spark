
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class analyse_SVM {


    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "D:\\hadoop-common-2.2.0-bin-master");



        SparkSession sparkSession=SparkSession.builder().appName("Mezuniyet_tezi").master("local").getOrCreate();
        Dataset<Row> raw_data = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("C:\\Users\\osman.sari\\Desktop\\heart_disease_dataset.csv");

        String[] headerList={"age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"};

        long startTime = System.currentTimeMillis();
        List<String> headers = Arrays.asList(headerList);
        List<String> headersResult=new ArrayList<>();
        for(String h:headers){

            if(h.equals("num")){
                StringIndexer indexTmp=new StringIndexer().setInputCol(h).setOutputCol("label");
                raw_data = indexTmp.fit(raw_data).transform(raw_data);
                headersResult.add("label");
            }
            else{
                StringIndexer indexTmp=new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase()+"_cat");
                raw_data = indexTmp.fit(raw_data).transform(raw_data);
                headersResult.add(h.toLowerCase()+"_cat");
            }


        }

        String[] colList=headersResult.toArray(new String[headersResult.size()]);
        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(colList).setOutputCol("features");

        Dataset<Row> transform_data = vectorAssembler.transform(raw_data);

        Dataset<Row> final_data = transform_data.select("label", "features");



        Dataset<Row>[] datasets = final_data.randomSplit(new double[]{0.80, 0.20});
        Dataset<Row> train_data = datasets[0];
        Dataset<Row> test_data = datasets[1];

        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(500)
                .setRegParam(5);

// Destek Vektör Makinesi
        LinearSVCModel lsvcModel = lsvc.fit(train_data);

        Dataset<Row> predictions = lsvcModel.transform(test_data);

        MulticlassClassificationEvaluator evulator=new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double evaluate = evulator.evaluate(predictions);



        System.out.println("oran: "+evaluate);
        // System.out.println("hatalı tahmin edilen veri sayısı:"+(1-evaluate)*66.66);

        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        System.out.println("sistem çalışma hızı: "+elapsedTime+" ms");


predictions.show();
    }}
