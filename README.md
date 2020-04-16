# CMPT733_Project

This project has four major components:

* a tag prediction model
* a politeness analysis model
* web demo
* visulization


### Prerequisites

* python
* tensorflow 1.5
* pytorch

### Tag Prediction Model

#### Preprocessing

#### Training

#### Make Prediction

### Politness analysis Model

#### Preprocessing

#### Make Prediction

### Web Demo

Download model checkpoints from 

```
https://drive.google.com/file/d/1hFkjRpyF4zfROgxvtKdOSpn0FWpB0eja
```

Extract and put it under ```/webserver```, so that we have a structure looks like ```./webserver/model/ ```


Make sure you are in ```/webserver``` directory, run following command to start server

```
python app.py
```

Open browser and go to ```http://localhost:5000/```


#### Training: make a copy of bert-base-uncased folder in webserver/model/language, and put it into impoliteness detection -> model folder
