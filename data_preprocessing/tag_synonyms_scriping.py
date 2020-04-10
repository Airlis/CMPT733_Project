from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd

synonyms = []
targets = []

# 42 for so
for page in range(1, 6):
    #url = 'https://stackoverflow.com/tags/synonyms?page=' + str(page) + '&tab=newest&dir=descending&filter=active.html'
    url = 'https://anime.stackexchange.com/tags/synonyms?page=' + str(page) + '&tab=newest&dir=descending&filter=active'
    response = get(url)
    f = open('data_preprocessing/pages/' + str(page) + '.html', 'w+')
    f.write(response.text)
    f.close()

    f = open('data_preprocessing/pages/' + str(page) + '.html', 'r')
    text = f.read()
    html_soup = BeautifulSoup(text, 'html.parser')
    synonym_containers = html_soup.select('tr[class*="synonym-"]')
    for i in range(len(synonym_containers)):
        synonym_info = synonym_containers[i]

        synonym = synonym_info.findAll('td')[0].text.replace('\n', '')
        if '×' in synonym:
            synonym = synonym.split('×')[-2]
        synonyms.append(synonym)

        target = synonym_info.findAll('td')[1].text.replace('\n', '')
        if ' ' in target:
            target = target.split(' ')[-2]
        targets.append(target)
    
synonym_df = pd.DataFrame({'synonym': synonyms,
                             'target': targets,
                             })
synonym_df.to_csv('Part 3/anime_synonym.csv', index=False)