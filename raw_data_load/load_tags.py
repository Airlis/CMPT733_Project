import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import re
from datetime import datetime
from pyspark.sql import SparkSession, functions, types, Row

cluster_seeds = ['199.60.17.32', '199.60.17.65']
 
spark = SparkSession.builder.appName('Spark Cassandra example').config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def load_badges(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Badges.xml')
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_UserId","userid")
    df = df.withColumnRenamed("_Name","name")
    df = df.drop('_Class', '_Date', '_TagBased')
    table_name = society + '_badges'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stack').save()

def load_posts(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Posts.xml').select("_Id", "_PostTypeId", "_ParentID", "_AcceptedAnswerId", "_Score", "_ViewCount", "_Tags", "_FavoriteCount", "_Title")
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_PostTypeId","posttypeid")
    df = df.withColumnRenamed("_ParentID","parentid")
    df = df.withColumnRenamed("_AcceptedAnswerId","acceptedanswerid")
    df = df.withColumnRenamed("_Score","score")
    df = df.withColumnRenamed("_ViewCount","viewcount")
    df = df.withColumnRenamed("_Tags","tags")
    df = df.withColumnRenamed("_FavoriteCount","favoritecount")
    df = df.withColumnRenamed("_Title","title")
    table_name = society + '_posts'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stack').save()

def load_tags(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Tags.xml').select("_Id", "_TagName", "_Count")
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_TagName","tagname")
    df = df.withColumnRenamed("_Count","count")
    table_name = society + '_tags'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stack').save()

def load_users(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Users.xml').select("_Id","_Location", "_Reputation", "_CreationDate")
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_Location","location")
    df = df.withColumnRenamed("_Reputation","reputation")
    df = df.withColumn('creationdate', functions.date_trunc('day', df['_CreationDate']))
    df = df.drop("_CreationDate")
    table_name = society + '_users'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stack').save()

def load_votes(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Votes.xml')
    df = df.withColumnRenamed("_Id", "id")
    df = df.withColumnRenamed("_PostId","postid")
    df = df.withColumnRenamed("_VoteTypeId","votetypeid")
    df = df.withColumnRenamed("_CreationDate","creationdate")
    df = df.withColumnRenamed("_UserId","userid")
    df = df.drop("_BountyAmount")
    df = df.where((df.votetypeid==2) | (df.votetypeid==3) | (df.votetypeid==5))
    table_name = society + '_votes'
    df.write.format("org.apache.spark.sql.cassandra").options(table=table_name, keyspace='stack').save()

def main(society):
    load_tags(society)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('Successfully loaded tags!')
 
if __name__ == '__main__':
    society = sys.argv[1]
    main(society)