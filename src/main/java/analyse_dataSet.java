import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.soundex;

public class analyse_dataSet {

    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "D:\\hadoop-common-2.2.0-bin-master");



        SparkSession sparkSession=SparkSession.builder().appName("Mezuniyet_tezi").master("local").getOrCreate();
        Dataset<Row> raw_data = sparkSession.read().format("csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("C:\\Users\\osman.sari\\Desktop\\heart_disease_dataset.csv");
/*
        raw_data.createOrReplaceTempView("heart");



        Dataset<Row> sex = sparkSession.sql("select sex from heart ");
        Dataset<Row> age = sparkSession.sql("select age from heart ");
        Dataset<Row> cp = sparkSession.sql("select cp from heart ");
        Dataset<Row> trestbps = sparkSession.sql("select trestbps from heart ");
        Dataset<Row> chol = sparkSession.sql("select chol from heart ");
        Dataset<Row> fbs = sparkSession.sql("select fbs from heart ");
        Dataset<Row> restecg = sparkSession.sql("select restecg from heart ");
        Dataset<Row> thalach = sparkSession.sql("select thalach from heart ");
        Dataset<Row> exang = sparkSession.sql("select exang from heart ");
        Dataset<Row> oldpeak = sparkSession.sql("select oldpeak from heart ");
        Dataset<Row> slope = sparkSession.sql("select slope from heart ");
        Dataset<Row> ca = sparkSession.sql("select ca from heart ");
        Dataset<Row> thal = sparkSession.sql("select thal from heart ");
        Dataset<Row> num = sparkSession.sql("select num from heart ");





        System.out.println(" cinsiyet :"+ sex.count()+"\n"+" yaş: "+age.count()+"\n"+" cp: "+cp.count()+"\n"+" trestbps: "
                +trestbps.count()+"\n"+" chol: "+chol.count()+"\n"+" fbs"+fbs.count()
                +"\n"+" restecg: "+restecg.count()+"\n"+" thalac: "+thalach.count()+"\n"+" exang: "+exang.count()+"\n"+" oldpeak: "
                +oldpeak.count()+"\n"+" slope: "+slope.count()+"\n"+" ca :"+ca.count()+"\n"+" thal: "+thal.count()+"\n"+" num: "+num.count());


*/



        Dataset<Row> new_dataFemale = raw_data.filter(new Column("sex").equalTo("0"));



        Dataset<Row> new_dataMale = raw_data.filter(new Column("sex").equalTo("1"));

        Dataset<Row> new_dataFemaleHeart = new_dataFemale.filter(new Column("num").equalTo("0"));

        Dataset<Row> new_dataMaleHeart = new_dataMale.filter(new Column("num").equalTo("0"));

        int kadin_sayisi= (int) new_dataFemale.count();
        int erkek_sayisi= (int) new_dataMale.count();
        int hasta_kadin= (int) new_dataFemaleHeart.count();
        int hasta_erkek= (int) new_dataMaleHeart.count();


        System.out.println("Kadın hasta sayısı: "+kadin_sayisi);
        System.out.println("Erkek hasta sayısı: "+erkek_sayisi);
        System.out.println("------------------------");
        System.out.println("Kalp hastası olan kadın sayısı: "+hasta_kadin);
        System.out.println("Kalp hastası olan erkek sayısı: "+hasta_erkek);

raw_data.show();
/*
        Dataset<Row> new_dataMale = raw_data.filter(new Column("sex"));
        int erkek_sayisi= (int) new_dataMale.count();

        System.out.println(erkek_sayisi);

 */
    }
}
