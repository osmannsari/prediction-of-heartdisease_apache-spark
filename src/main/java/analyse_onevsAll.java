import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import shapeless.the;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.Media.print;


public class analyse_onevsAll {





    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "D:\\hadoop-common-2.2.0-bin-master");
        long startTime = System.currentTimeMillis();

        SparkSession sparkSession=SparkSession.builder().appName("Mezuniyet_tezi").master("local").getOrCreate();
        Dataset<Row> raw_data = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("C:\\Users\\osman.sari\\Desktop\\heart.csv");
        // Split the data into train and test

        String[] headerList={"age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"};

        List<String> headers = Arrays.asList(headerList);
        List<String> headersResult=new ArrayList<>();
        for(String h:headers){

            if(h.equals("target")){
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
        Dataset<Row> train = datasets[0];
        Dataset<Row> test = datasets[1];


        // configure the base classifier.
        LogisticRegression classifier = new LogisticRegression()
                .setMaxIter(10)
                .setTol(0x3e8)
                .setFitIntercept(true);


        OneVsRest ovr = new OneVsRest().setClassifier(classifier);

//OnevsAll
        OneVsRestModel ovrModel = ovr.fit(train);


        Dataset<Row> predictions = ovrModel.transform(test)
                .select("prediction", "label");

        predictions.show();




        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy");


        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Oran = " +accuracy);

       System.out.println("hatalı tahmin sayısı:"+ (1-accuracy)*66.66);



        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
        System.out.println("sistem çalışma hızı: "+elapsedTime+" ms");



    }
}