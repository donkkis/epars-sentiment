# Sentiment Analysis from EPAR documents

Opinion mining from European Public Assesment Reports.

## Setup for local dev/test purposes (Windows 10 / Powershell)

Recommended configuration is to use ```python>=3.7.6``` together with ```virtualenv```. Any reasonably new Python version should work, though.

1. In project root, run ```virtualenv .venv```
2. Activate your new virtualenv: ```.venv\Scripts\activate```
3. ```pip install -r requirements.txt```
4. Create a file called `.env` in project root and configure there pointer to your raw data file, e.g.:

```DEFAULT_DATAPATH = 'C:/Users/panaho/epars-sentiment/data/sentences_with_sentiment.xlsx'```

## (Optional) setup experiment tracking with mlflow

Put in ```.env``` file the URI where you wish to write mlflow logs and artifacts, e.g.:

```MLFLOW_TRACKING_URI = 'file://C:/Users/panaho/epars-sentiment/mlruns'```

Here it is assumed that runs are logged locally to plain files. For other logging options, see <a href='https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded'>mlflow docs</a>.

Alternatively, experiments can be run using the ```--no_log``` flag to omit mlflow logging.

## Running experiments

An example experiment ```bow.py``` is provided in ```experiments``` directory. Use the ```--no_log``` if you did not configure mlflow logging.

## Report

A short presentation outlining the findings in preliminary study can be found under the ```report``` directory