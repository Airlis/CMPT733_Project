import xml.etree.ElementTree as ET
import pandas as pd

def load_from_xml():
    root = ET.parse('Posts.xml').getroot()
    tags = {"tags":[]}
    for elem in root:
        tag = {}
        tag["Id"] = elem.attrib['Id']
        tag["PostTypeId"] = elem.attrib['PostTypeId']
        if tag["PostTypeId"] != '1':
            continue
        tag["CreationDate"] = elem.attrib['CreationDate']
        tag["Tags"] = elem.attrib['Tags']
        tags["tags"]. append(tag)

    df_posts = pd.DataFrame(tags["tags"])
    df_posts.to_csv('posts_for_eda.csv', index=False)


def get_tags_and_count():
    df_posts = pd.read_csv('posts_for_eda.csv')  
    df_posts['CreationDate'] =  pd.to_datetime(df_posts['CreationDate'], format="%Y-%m-%d")
    df_posts['Year'] = df_posts['CreationDate'].dt.year
    df_posts_stable = df_posts[df_posts['Year'] <= 2017].drop(['CreationDate', 'PostTypeId'], axis=1)
    df_posts_stable['Tags'] = df_posts_stable['Tags'].str.replace('<', '').str.split('>')
    df_posts_stable['TagCounts'] = df_posts_stable['Tags'].str.len() - 1
    df_posts_stable.to_csv('posts_for_eda_cleaned.csv', index=False)

get_tags_and_count()




