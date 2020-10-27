# Insight - RecNet
Recommender System Powered by Deep Neural Network

## Motivation
- Build a flexible and extensible Recommender System using TensorFlow 2.0
- Deploy the Recommender app on AWS in Django framework
- The app makes REST API call to TensorFlow Serving with Docker
- Benchmark the system against another Machine Learning model

## Requirements
- Python 3.x
- TensorFlow 2.x
- Scikit-Learn
- Scikit-Surprise
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

## Run - Command Line
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
[On YouTube](https://www.youtube.com/watch?v=XMaNK1GZjzw)
