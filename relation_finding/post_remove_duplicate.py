import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def main():
    df = spark.read.format("csv").option("header", "true").load('so_tags1.csv')
    df.show(repr)

    df = df.dropDuplicates()
    
    df.write.csv('so_tags.csv',header = 'true')

if __name__ == '__main__':
    main()