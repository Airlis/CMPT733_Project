import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from datetime import datetime
from pyspark.sql import SparkSession, functions, types, Row
from pyspark.sql.functions import udf

cluster_seeds = ['199.60.17.32', '199.60.17.65']

spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).config('spark.dynamicAllocation.maxExecutors', 16).getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def main(society, origin):
    df_cas = spark.read.format("org.apache.spark.sql.cassandra").options(table=society, keyspace='stack').load()
    print('Finished loading Cassandra DF')

    df_ori = spark.read.format('xml').options(rowTag='row').load(origin + '.xml')
    df_ori = df_ori.withColumnRenamed("_Id", "id")
    df_ori = df_ori.withColumnRenamed("_PostId","postid")
    df_ori = df_ori.withColumnRenamed("_VoteTypeId","votetypeid")
    df_ori = df_ori.withColumnRenamed("_CreationDate","creationdate")
    df_ori = df_ori.withColumnRenamed("_UserId","userid")
    df_ori = df_ori.drop("_BountyAmount")
    df_ori = df_ori.where((df_ori.votetypeid==2) | (df_ori.votetypeid==3) | (df_ori.votetypeid==5))
    print('Finished loading Original DF')

    df = df_ori.subtract(df_cas)
    print('Diff in rows:')
    print(df.count())

    df.write.format("org.apache.spark.sql.cassandra").mode('append').options(table=society, keyspace='stack').save()


if __name__ == '__main__':
    society = sys.argv[1]
    origin = sys.argv[2]
    main(society, origin)