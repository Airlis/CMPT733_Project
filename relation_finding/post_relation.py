import sys
from pyspark.sql import SparkSession, functions, types, Row, Window

spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def relation(society_1, society_2, tag, df_society_1, df_society_2):
    df_society_1_with_tag = df_society_1.where(df_society_1[str(society_1 + "_tags")] == tag).drop(str(society_1 + "_tags")).drop(str(society_1 + "_ownerId")).dropDuplicates()

    df_society_2_selected = df_society_2.join(df_society_1_with_tag, on = ['accountId'], how = 'inner')

    df_result = df_society_2_selected.groupby(str(society_2 + "_tags")).count().sort(functions.desc("count"))
    df_result = df_result.withColumnRenamed('count', str(society_2 + "_count"))
    df_result.show()

    df_result.write.csv(society_2 + '_' + tag + '.csv',header = 'true')

    return df_result


def main(society_1, society_2, society_3, society_4, society_5, tag):
    df_society_1 = spark.read.format("csv").option("header", "true").load(society_1 + '_tags.csv')
    df_society_1.show()
    df_society_2 = spark.read.format("csv").option("header", "true").load(society_2 + '_tags.csv')
    df_society_2.show()
    df_society_3 = spark.read.format("csv").option("header", "true").load(society_3 + '_tags.csv')
    df_society_3.show()
    df_society_4 = spark.read.format("csv").option("header", "true").load(society_4 + '_tags.csv')
    df_society_4.show()
    df_society_5 = spark.read.format("csv").option("header", "true").load(society_5 + '_tags.csv')
    df_society_5.show()

    df_result_1 = relation(society_1, society_2, tag, df_society_1, df_society_2)
    df_result_2 = relation(society_1, society_3, tag, df_society_1, df_society_3)
    df_result_3 = relation(society_1, society_4, tag, df_society_1, df_society_4)
    df_result_4 = relation(society_1, society_5, tag, df_society_1, df_society_5)

    # df_result_1 = spark.read.format("csv").option("header", "true").load(society_2 + '_' + tag + '.csv').sort(functions.desc(str(society_2 + "_count")))
    # df_result_1.show()
    # df_result_2 = spark.read.format("csv").option("header", "true").load(society_3 + '_' + tag + '.csv').sort(functions.desc(str(society_3 + "_count")))
    # df_result_3 = spark.read.format("csv").option("header", "true").load(society_4 + '_' + tag + '.csv').sort(functions.desc(str(society_4 + "_count")))
    # df_result_4 = spark.read.format("csv").option("header", "true").load(society_5 + '_' + tag + '.csv').sort(functions.desc(str(society_5 + "_count")))

    window = Window.orderBy(functions.col(str(society_2 + "_count")).desc())
    df_result_1 = df_result_1.withColumn('id', functions.row_number().over(window))
    df_result_1.show()
    window = Window.orderBy(functions.col(str(society_3 + "_count")).desc())
    df_result_2 = df_result_2.withColumn('id', functions.row_number().over(window))
    df_result_2.show()
    window = Window.orderBy(functions.col(str(society_4 + "_count")).desc())
    df_result_3 = df_result_3.withColumn('id', functions.row_number().over(window))
    df_result_3.show()
    window = Window.orderBy(functions.col(str(society_5 + "_count")).desc())
    df_result_4 = df_result_4.withColumn('id', functions.row_number().over(window))
    df_result_4.show()

    df_join = df_result_1.join(df_result_2, on = ['id'], how = 'outer').sort(functions.asc("id"))
    df_join.show()
    df_join = df_join.join(df_result_3, on = ['id'], how = 'outer').sort(functions.asc("id"))
    df_join.show()
    df_join = df_join.join(df_result_4, on = ['id'], how = 'outer').sort(functions.asc("id"))

    df_join.show()

    df_join.write.csv('tag_' + tag + '.csv',header = 'true')

    

if __name__ == '__main__':
    society_1 = sys.argv[1]
    society_2 = sys.argv[2]
    society_3 = sys.argv[3]
    society_4 = sys.argv[4]
    society_5 = sys.argv[5]
    tag = sys.argv[6]
    main(society_1, society_2, society_3, society_4, society_5, tag)