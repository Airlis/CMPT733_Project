import xml.etree.ElementTree as ET
import pandas as pd

def load_from_xml():
    root = ET.parse('PostHistory.xml').getroot()
    tags = {"tags":[]}
    for elem in root:
        tag = {}
        tag["Id"] = elem.attrib['Id']
        tag["PostHistoryTypeId"] = elem.attrib['PostHistoryTypeId']
        if tag["PostHistoryTypeId"] != '6':
            continue
        tag["PostId"] = elem.attrib['PostId']
        tags["tags"]. append(tag)

    df_posts = pd.DataFrame(tags["tags"])
    df_posts.to_csv('posthistory_for_eda.csv', index=False)

analysis()