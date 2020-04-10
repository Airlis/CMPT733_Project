import pandas as pd


def posts_analysis():
    df_posts = pd.read_csv('posts_for_eda_cleaned.csv')
    num_rows = len(df_posts.index)
    num_tags_all = df_posts['TagCounts'].sum()

    df_year = df_posts.groupby(['Year']).mean().drop(['Id'], axis=1)
    print(df_year)

    print('Number of posts by the end of 2017: ' + str(num_rows))
    print('Number of tags of all posts by the end of 2017: ' + str(num_tags_all))
    print('Average number of tags of all posts by the end of 2017: ' + str(num_tags_all / num_rows))

def history_analysis():
    df_posts = pd.read_csv('posts_for_eda_cleaned.csv').rename(columns={"Id": "PostId"})
    df_posthistory = pd.read_csv('posthistory_for_eda.csv').drop(['Id'], axis=1)


    num_rows = len(df_posthistory.index)
    
    df_count = df_posthistory.groupby(['PostId']).size().reset_index().rename(columns={0: "Number of Update"})

    df_posthistory_year = pd.merge(df_posts, df_count, on='PostId', how='inner')


    print('Number of changes in tags: ' + str(num_rows))
    print(df_count.max())
    print(df_posthistory_year)

    df_change = df_count.groupby(['Number of Update']).size()
    print(df_change)


posts_analysis()