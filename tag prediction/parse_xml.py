import xml.etree.ElementTree as ET
import pandas as pd
import re
import os
from tqdm import tqdm

def main(datasets, tag_threshold=50):
    for dataset in datasets:
        path = os.path.join('community-data',dataset)
        folder = os.path.join('community-data',dataset , str(tag_threshold))
        if not os.path.exists(folder):
            os.makedirs(folder)

        # load tag 
        tags_df = pd.DataFrame(columns=['TagName', 'Count'])
        tags_tree = ET.parse(os.path.join(path, 'Tags.xml'))
        for tag in tags_tree.findall('row'):
            tag_name = tag.get('TagName')
            tag_count = tag.get('Count')
            if tag_name is not None and tag_count is not None:
                tags_df = tags_df.append(
                    {'TagName': tag_name, 'Count': int(tag_count)}, ignore_index=True)

        # load posts 
        posts_df = pd.DataFrame(columns=['Id', 'Text', 'Tags'])
        posts_tree = ET.parse(os.path.join(path,'Posts.xml'))

        print("Processing [" + dataset + "]")
        for row in tqdm(posts_tree.findall('row')):
            post_id = row.get('Id')
            body = row.get('Body')
            body = re.sub('<.+?>', '', body)
            body = re.sub('\s+', ' ', body).strip()
            title = row.get('Title')
            text = ''
            if title is not None:
                text = title
            if body is not None:
                text += (' ' + body)
            if text == '':
                continue
            tags = row.get('Tags')
            if tags is not None:
                tags = re.findall('<(.+?)>', tags)
                tags = tags_df['TagName'][
                    (tags_df['TagName'].isin(tags)) & (tags_df['Count'] >= tag_threshold)].tolist()
            else:
                continue
            # print('{},{},{},{}'.format(post_id, body, title, tags))
            if len(tags) != 0:
                # print(tags.tolist())
                # write_to_file(folder, post_id, text, tags)
                posts_df = posts_df.append({'Id': post_id, 'Text': text,
                                            'Tags': tags}, ignore_index=True)

        posts_df.to_csv(os.path.join(folder,'posts_df.csv'), encoding='utf8')
        tags_df[tags_df['Count']>=tag_threshold].to_csv(os.path.join(folder,'tags_df.csv'), encoding='utf8')


if __name__ == '__main__':
    # datasets = ['askubuntu', 'codereview', 'unix', 'serverfault']
    datasets = ['gardening']
    tag_threshold = [50]
    for t in tag_threshold:
        main(datasets, t)
