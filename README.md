# CMPT733_Project

This project has four major components:

* a tag prediction model
* a politeness analysis model
* visulization
* web demo


### Prerequisites

* python
* tensorflow 1.15
* pytorch
* pyspark 2.4

### Tag Prediction Model

To reproduce the training result, download Stack Exchange data dump, run ```parse_xml.py``` to preprocess the dataset, then run training script ```train.py```

### Offensive Language Detection Model

To reproduce the training result, make a copy of bert-base-uncased folder in webserver/model/language, and put it into ```impoliteness detection/model```  folder, and run ```train.py```.

### Cross-Platforms Analysis

#### Preprocessing

The dataset is downloaded from https://archive.org/details/stackexchange


Please download the files related to StackOverflow, Bicycle, Games, Movie and Music and upload them to the cluster of Big Data program. The name of the files should be renamed as following:  ``so_Posts.xml``, ``so_Users.xml``, ``bicycle_Posts.xml``, etc.

The files foe EDA are ``posthistory_tag_only.py``, ``posts_time.py``, ``tag_history_analysis.py``. Please run with python command to view the results.

#### Tag Synonyms

Please download BeautifulSoup before running the code.

The file to run tag synonyms is ``tag_synonyms_scriping.py``. 

Please edit the file to have the right web page and page number (for the for loop) to do scriping. The site is the "Tag Synonym" part of each society in Stack Exchange. The results are saved as csvs.

``tag_synonym_evaluation.py`` checks the results.

#### Find relations

First run ``spark-submit --packages com.databricks:spark-xml_2.11:0.7.0 user_matching.py so bicycle games movie music`` to get all the cross-platform users.

Run ``spark-submit --packages com.databricks:spark-xml_2.11:0.7.0 post_tag_exploding.py so`` to explode the tags for each users by given posts. Replace ``so`` to  ``bicycle`` ``games`` ``movie`` ``music`` to do the same things for all the societies.

Run ``spark-submit post_relation.py so bicycle games movie music [tag]`` to get the behaviour of users by a given tag in stackoverflow. Please change ``[tag]`` to any tags you want to do the analysis.

Finally run ``spark-submit [tag]`` to see the statistcal results of the provided tags with the five given societies.

### Web Demo

Download model checkpoints from 

https://drive.google.com/file/d/1hFkjRpyF4zfROgxvtKdOSpn0FWpB0eja

Extract and put it under ```/webserver```, so that we have a structure looks like ```./webserver/model/ ```


Make sure you are in ```/webserver``` directory, run following command to start server

```
python app.py
```

Open browser and go to ```http://localhost:5000/```

