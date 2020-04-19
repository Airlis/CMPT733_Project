import sys
from pyspark.sql import SparkSession, functions, types, Row, Window

spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

def relation(society_1, society_2, tag, df_society_1, df_society_2, so_count):
    df_society_1_with_tag = df_society_1.drop(str(society_1 + "_tags")).drop(str(society_1 + "_ownerId"))

    df_society_2_selected = df_society_2.join(df_society_1_with_tag, on = ['accountId'], how = 'inner').dropDuplicates()

    count = df_society_2_selected.select('accountId').dropDuplicates().count()
    print('Number of ' + society_2 + ' user with ' + tag + ': ' + str(count))
    print('Percentage of ' + society_2 + ' user with ' + tag + ': ' + str(count / so_count * 100))

    df_result = df_society_2_selected.groupby(str(society_2 + "_tags")).count().sort(functions.desc("count"))
    df_result = df_result.withColumnRenamed('count', str(society_2 + "_count"))
    df_result = df_result.withColumn(str(society_2 + "_percent"), df_result[str(society_2 + "_count")] / count * 100)
    df_result.show()

    return df_result

def main(tag):
    so_count = 128167
    df_joined_account = spark.read.format("csv").option("header", "true").load('joined_user.csv')
    so_count = df_joined_account.count()
    print('Number of stack overflow user: ' + str(so_count))

    df_bicycle_account = spark.read.format("csv").option("header", "true").load('joined_user1.csv')
    bicycle_count = df_bicycle_account.count()
    print('Number of bicycle user: ' + str(bicycle_count))

    df_game_account = spark.read.format("csv").option("header", "true").load('joined_user2.csv')
    game_count = df_game_account.count()
    print('Number of game user: ' + str(game_count))

    df_movie_account = spark.read.format("csv").option("header", "true").load('joined_user3.csv')
    movie_count = df_movie_account.count()
    print('Number of movie user: ' + str(movie_count))

    df_music_account = spark.read.format("csv").option("header", "true").load('joined_user4.csv')
    music_count = df_music_account.count()
    print('Number of music user: ' + str(music_count))

    print('---------------------------------------------------------------')

    print('Percentage of bicycle user: ' + str(bicycle_count / so_count * 100))

    print('Percentage of game user: ' + str(game_count / so_count * 100))

    print('Percentage of movie user: ' + str(movie_count / so_count * 100))

    print('Percentage of music user: ' + str(music_count / so_count * 100))


    df_society_1 = spark.read.format("csv").option("header", "true").load('so_tags.csv')
    df_society_2 = spark.read.format("csv").option("header", "true").load('bicycle_tags.csv')
    df_society_3 = spark.read.format("csv").option("header", "true").load('game_tags.csv')
    df_society_4 = spark.read.format("csv").option("header", "true").load('movie_tags.csv')
    df_society_5 = spark.read.format("csv").option("header", "true").load('music_tags.csv')

    df_so_tag = df_society_1.filter(df_society_1['so_tags'] == tag).dropDuplicates()
    count_so_tag = df_so_tag.count()
    print('Number of ' + tag + ' user: ' + str(count_so_tag))
    print('Percentage of ' + tag + ' user: ' + str(count_so_tag / so_count * 100))

    df_result_1 = relation('so', 'bicycle', tag, df_so_tag, df_society_2, count_so_tag)
    df_result_2 = relation('so', 'game', tag, df_so_tag, df_society_3, count_so_tag)
    df_result_3 = relation('so', 'movie', tag, df_so_tag, df_society_4, count_so_tag)
    df_result_4 = relation('so', 'music', tag, df_so_tag, df_society_5, count_so_tag)



if __name__ == '__main__':
    tag = sys.argv[1]
    main(tag)