# Insight - RecNet
Recommender System Powered by Deep Neural Network

## Motivation
- Build a Recommender System using TensorFlow 2.0
- Deploy the Recommender app on AWS in Django framework
- Benchmark the system against other Deep Neural Network model and Machine Learning model

## Requirements
- Python 3.x
- TensorFlow 2.x
- Scikit-Learn
- Pandas
- Numpy

## Data
- [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
  - [Book (ratings only)](http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv)
  - [Book (metadata)](https://forms.gle/A8hBfPxKkKGFCP238)

## Setup
- Clone repository: `git clone https://github.com/jt2002/Insight.git`
- Download data from [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html)
- Rename `Books.csv` to `Book_review_rating.csv`
- Move the CSV file to the folder `data\source_data`
- Gunzip `meta_Books.json.gz` to `meta_Books.json`
- Move the JSON file to the folder `data\source_data`

## Run on Command Line
Preprocess data
```
python preprocess.py
```

Build and save the Deep Neural Network model
```
python build_save_model.py
```

Generate recommendation
```
python recnet.py
```

## Live Application
[RecNet](http://www.recnet.xyz/)
