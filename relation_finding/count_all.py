import sys
from pyspark.sql import SparkSession, functions, types, Row, Window

spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def load_users(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Users.xml')
    print(society + ': ' + str(df.count()))

def main():
    load_users('so')
    load_users('bicycle')
    load_users('game')
    load_users('movie')
    load_users('music')

if __name__ == '__main__':
    main()