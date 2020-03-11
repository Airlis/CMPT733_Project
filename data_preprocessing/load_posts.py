import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re
from datetime import datetime
from pyspark.sql import SparkSession, functions, types, Row

cluster_seeds = ['199.60.17.32', '199.60.17.65']
 
spark = SparkSession.builder.appName('Spark Cassandra StackOverflow').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext


def load_posts(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Posts.xml').select("_Id", "_PostTypeId", "_Body", "_Tags", "_Title", "_OwnerUserId")
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_PostTypeId","posttypeid")
    df = df.withColumnRenamed("_Body","body")
    df = df.withColumnRenamed("_Tags","tags")
    df = df.withColumnRenamed("_Title","title")
    df = df.withColumnRenamed("_OwnerUserId","userid")

    

    df.show()
    table_name = society + '_posts'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stackexchange').save()


def main(society):
    load_posts(society)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Successfully loaded posts!')
 
if __name__ == '__main__':
    society = sys.argv[1]
    main(society)