import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def load_users(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Users.xml').select("_Id","_AccountId")
    df = df.withColumnRenamed("_Id", str(society + "_userId"))
    df = df.withColumnRenamed("_AccountId", str(society + "_accountId"))
    return df

def join(society_1, society_2, df_society_1, df_society_2):
    
    df_join = df_society_1.join(df_society_2, df_society_1[str(society_1 + "_accountId")] == df_society_2[str(society_2 + "_accountId")], 'inner')

    df_join = df_join.drop(str(society_1 + "_accountId"))
    df_join = df_join.withColumnRenamed(str(society_2 + "_accountId"), "accountId")
    df_join.show()

    return df_join

def main(society_1, society_2, society_3, society_4, society_5):
    # df_society_1 = load_users(society_1)
    # df_society_2 = load_users(society_2)
    # df_society_3 = load_users(society_3)
    # df_society_4 = load_users(society_4)
    # df_society_5 = load_users(society_5)

    # df_join1 = join(society_1, society_2, df_society_1, df_society_2)
    # print(df_join1.count())
    # df_join2 = join(society_1, society_3, df_society_1, df_society_3)
    # print(df_join2.count())
    # df_join3 = join(society_1, society_4, df_society_1, df_society_4)
    # print(df_join3.count())
    # df_join4 = join(society_1, society_5, df_society_1, df_society_5)
    # print(df_join4.count())

    # df_join1.write.csv('joined_user1.csv',header = 'true')
    # df_join2.write.csv('joined_user2.csv',header = 'true')
    # df_join3.write.csv('joined_user3.csv',header = 'true')
    # df_join4.write.csv('joined_user4.csv',header = 'true')

    df_join1 = spark.read.format("csv").option("header", "true").load('joined_user1.csv')
    df_join2 = spark.read.format("csv").option("header", "true").load('joined_user2.csv')
    df_join3 = spark.read.format("csv").option("header", "true").load('joined_user3.csv')
    df_join4 = spark.read.format("csv").option("header", "true").load('joined_user4.csv')
    
    df_join = df_join1.join(df_join2, on = [str(society_1 + "_userId"), 'accountId'], how = 'outer')
    df_join = df_join.join(df_join3, on = [str(society_1 + "_userId"), 'accountId'], how = 'outer')
    df_join = df_join.join(df_join4, on = [str(society_1 + "_userId"), 'accountId'], how = 'outer')
    df_join.show()
    print(df_join.count())
    df_join.write.csv('joined_user.csv',header = 'true')

 
if __name__ == '__main__':
    society_1 = sys.argv[1]
    society_2 = sys.argv[2]
    society_3 = sys.argv[3]
    society_4 = sys.argv[4]
    society_5 = sys.argv[5]
    main(society_1, society_2, society_3, society_4, society_5)