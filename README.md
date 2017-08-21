# Numerai Model

[![Datmo Model](https://beta.datmo.io/shabazp/numerai-model/badge.svg)](https://beta.datmo.io/shabazp/numerai-model)

[Numer.ai](https://numer.ai/) is a hedge fund that hosts weekly online competitions for data science model building, allowing data science enthuasists a way to earn money by building models that help to explain phenomena in their sanitized data. A great part about working with Numer.ai data is that it’s already curated for you — no need to deal with wrangling, scraping, or cleaning. This is pretty awesome, as data cleaning is one of the more frustrating/boring parts of machine learning.

This is a [Datmo](https://datmo.com) model which is built over numerai dataset.

## Approach:

The jupyter notebook explains different steps taken while building the model.

1. We perform an exploratory data analysis (EDA) to visualize and explore the feature space. We made specific inference of features having strong positive and negative correlation.

2. We built an XGBoost model over the whole dataset. This gave us a better understanding of important features and provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.

3. We use this importance score to perform dimensionality reduction and finally we come down from 21 to 4 features. This improved the accuracy. 

4. The last section of code uses a dictionary, which is a possible input for API and prediction is performed using the built model on that data.

There are still many possibilities of improving the performance of the model by having grid search over all possiblities of XGBoost models.

## Datmo Model Description:

Check out the Snapshots tab above for information on snapshots that were taken with various parameters. Search and filter them to find the best ones. 

(Coming soon) Click on the "Deploy" button next to the relevant snapshot to create an API on the server the Agent is running on or use the deploy command in the CLI to deploy it as a RESTful API. 

Clone this model on your local machine with the Datmo CLI

```
$ datmo clone https://beta.datmo.io/shabazp/numerai-model
```

Now you can run the set of commands below to better understand the advantages of converting a repository to a Datmo model. 
First you can check out all of the snapshots that have already been created by the user. 

```
$ datmo snapshot ls 
```

Once you have viewed all of the existing snapshots, you can create your own by simply running the code below which runs the regression training which saves a few key files (the weights file, metrics for this file, and ) in the output directory which creates a new snapshot. 

```
$ datmo task run "jupyter notebook" --port 8888
```

For reference, here are few more quick details about what you can find in the repository. 

```
Dockerfile: this is meant to keep track of your environment. For example, this Dockerfile ensures that our environment has all of the requirements needed to run our model. You can manage, edit, and create new enviroments with the datmo env command. Each session you create with datmo will have a default environment associated with it.

_datmo: visible datmo folder where the input information (data, snapshots, misc files) will go.

_datmo/data: we store any data that you might use for the model. In this case, we are pulling data from a remote server and will populate this directory automatically while training. For other models, you will want to use the datmo dataset command to create, manage, or pull datasets from the remote server

_datmo/snapshots: contains weights, statistics and files(visualizations) that are pulled in to be inputs for processing
```
The commands and quick reference above is by no means meant to be comprehensive. Please refer to the [CLI documentation](https://beta-docs.datmo.io) for more details on the various Datmo commands to use and file structures present in Datmo.

Happy Building :)
