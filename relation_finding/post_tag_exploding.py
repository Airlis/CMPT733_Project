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

def load_posts(society):
    df = spark.read.format('xml').options(rowTag='row').load(society + '_Posts.xml').select("_Id", "_PostTypeId", "_ParentID", "_Tags", "_OwnerUserId")
    df = df.withColumnRenamed("_Id", str(society + "_postId"))
    df = df.withColumnRenamed("_PostTypeId", str(society + "_typeId"))
    df = df.withColumnRenamed("_ParentID", str(society + "_parentID"))
    df = df.withColumnRenamed("_Tags", str(society + "_tags"))
    df = df.withColumnRenamed("_OwnerUserId", str(society + "_ownerId"))
    return df

def main(society):
    df_user_join = spark.read.format("csv").option("header", "true").load('joined_user.csv')
    df_user_join.show()

    df_posts_society = load_posts(society)
    df_posts_society_filterd = df_posts_society.join(df_user_join, df_posts_society[str(society + "_ownerId")] == df_user_join[str(society + "_userId")], 'inner').drop(str(society + "_userId"))
    df_posts_society_filterd.show()

    df_posts_society_question = df_posts_society_filterd.where(df_posts_society_filterd[str(society + "_typeId")] == 1)
    df_posts_society_answer = df_posts_society_filterd.where(df_posts_society_filterd[str(society + "_typeId")] == 2).drop(str(society + "_postId")).drop(str(society + "_typeId")).drop(str(society + "_tags"))
    df_posts_society_answer = df_posts_society_answer.withColumnRenamed(str(society + "_parentID"), str(society + "_answerParentId"))
    df_posts_society_answer = df_posts_society_answer.withColumnRenamed(str(society + "_ownerId"), str(society + "_answerOwnerId"))

    df_posts_society_question.show()
    df_posts_society_answer.show()

    df_posts_society_answer_post = df_posts_society.join(df_posts_society_answer, df_posts_society[str(society + "_postId")] == df_posts_society_answer[str(society + "_answerParentId")], 'inner')
    df_posts_society_answer_post.show()

    df_posts_society_question = df_posts_society_question.select(str(society + "_tags"), str(society + "_ownerId"), 'accountId')
    df_posts_society_question.show()
    print(df_posts_society_question.count())
    df_posts_society_answer_post = df_posts_society_answer_post.select(str(society + "_tags"), str(society + "_answerOwnerId"), 'accountId')
    df_posts_society_answer_post = df_posts_society_answer_post.withColumnRenamed(str(society + "_answerOwnerId"), str(society + "_ownerId"))
    df_posts_society_answer_post.show()
    print(df_posts_society_answer_post.count())
    df_posts_society_merge = df_posts_society_question.unionAll(df_posts_society_answer_post)
    df_posts_society_merge.show()
    print(df_posts_society_merge.count())
    df_posts_society_merge.write.csv(society + '_all_tags_combined.csv',header = 'true')

    # df_posts_society_merge = spark.read.format("csv").option("header", "true").load(society + '_all_tags_combined.csv')
    df_posts_society_merge.show()

    df_posts_society_merge = df_posts_society_merge.withColumn('tag',functions.explode(functions.array_remove(functions.split(str(society + "_tags"), r"(\<)|(\>)"), "")))
    df_posts_society_merge = df_posts_society_merge.drop(str(society + "_tags"))
    df_posts_society_merge = df_posts_society_merge.withColumnRenamed('tag', str(society + "_tags"))
    df_posts_society_merge.show()

    # uncommand these lines if tag synonyms are used
    # df_synonym = spark.read.format("csv").option("header", "true").load(society + '_synonym.csv')
    # df_synonym.show()

    # df_posts_society_merge = df_posts_society_merge.withColumn(str(society + "_tags"), functions.when(df_posts_society_merge[str(society + "_tags")] == df_synonym['synonym'], df_synonym['target']).otherwise(df_posts_society_merge[str(society + "_tags")]))

    # df_posts_society_merge.show()

    df_posts_society_merge.write.csv(society + '_tags.csv',header = 'true')

if __name__ == '__main__':
    society = sys.argv[1]
    main(society)