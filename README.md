# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Problem Statement
In this project a bank marketing [dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) is used.
It contains phone calls from a direct marketing compaign of a Portoguese banking institution.

The dataset has a series of information (age, job, marital, education, etc...) for a total of 32950 observations, 20 features, and a target variable (y)
with two possible values: yes or no.
The task is addressed as a classification task and the goal is to predict if a client will subscribe a term deposit (y variable).

Two different approaches have been investigated. The first one use a logistic regression model with hyperparameters tuning using HyperDrive,
the second one use the power of AutoML.

The main steps are reported in the diagram below:
![Steps](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/img/problem_statement_steps.png)

The [udacity-project.ipynb](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/udacity-project.ipynb) jupyter notebook contains all the steps of the entire procedure.

## Scikit-learn Pipeline
** Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm. **

In this first task, an HyperDrive pipeline using a scikit-learn logistic regression model is built.
Logistic regression is a classification algorithm used when the dipendent variable (y) is categorical. It uses the logistic function to model the probability of a certain class or event (yes or no).

The [train.py](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/train.py) script contains the following steps:

  - Load data as TabularDataset using TabularDatasetFactory
  - Transform and clean data: the clean_data() function is used for handle missing values, create dummies variables for categorical features, mapping values and other transformations in order to make data suitable for modeling
  - Split data into train and test set: 80% train, 20% test
  - Train the logistic regression model on training data with two hyperparameters:
  
    - *C*: inverse of regularization strength. Smaller values cause stronger regularization
    - *max_iter*: maximum number of iterations for model to converge
    
    The goal is to tune these two parameters using HyperDrive.

Hyperparameter tuning can be computationally expensive, so HyperDrive helps to automate and speeds up hyperparameter tuning process, choosing these parameters.
*HyperDriveConfig* class is responsable of the hyperparameters tuning process. It includes information about hyperparameter space sampling, termination policy, primary metric and estimator.

Specify hyperparameter space sampling and termination policy is very important:

  - Hyperparameter space sampling: ***RandomParameterSampling*** randomly select hyperparameters values over the search space.
    It is not computationally expensive and it is not exhaustive but it works well in most cases. *C* and *max_iter* parameters have been passed to the sampler:
  
    ```
    # Specify parameter sampler
    ps = RandomParameterSampling(
      {
          "--C": uniform(0.1, 1.0),
          "--max_iter": choice(25, 50, 100, 150)
      }
    )
    ```

  - Termination policy: ***BanditPolicy*** defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation.
    Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.
    It automatically terminates poorly performing runs, saving time and improving computational efficiency:
    
    ```
    # Specify a Policy
    policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
    ```

After submitting the hyperdrive run to the experiment, the best run and the related metrics have been collected:
```
best_run_hdr = hdr.get_best_run_by_primary_metric()
best_run_metrics_hdr = best_run_hdr.get_metrics()
best_params_hdr = best_run_hdr.get_details()['runDefinition']['arguments']

print('Best run ID: ', best_run_hdr.id)
print('Best run Accuracy: ', best_run_metrics_hdr['Accuracy'])
print('Metrics: ', best_run_metrics_hdr)
```
```
Best run ID: HD_d4628b90-0e0d-4602-b36f-903f7ea498ec_2
Best run Accuracy: 0.9072837632776934
Metrics: {'Regularization Strength:': 0.5056344015312062, 'Max iterations:': 25, 'Accuracy': 0.9072837632776934}
```
The **best model** has been stored [here](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/outputs/best_model_hyperdrive.joblib):
  - ***Parameters***: *C* =  0.5056344015312062, *max_iter* = 25
  - ***Accuracy***: 0.9072837632776934


## AutoML
In this second task, an AutoML pipeline is built.
Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows to build ML models with high scale, efficiency, and productivity all while sustaining model quality.

*AutoMLConfig* class is responsable of the automated machine learning process. It contains the parameters for configuring the experiment run:

```
# Set parameters for AutoMLConfig
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=train_data,
    label_column_name='y',
    n_cross_validations=4,
    compute_target=compute_cluster)
```

AutoML pipeline is performed following these step:

  - Load data as TabularDataset using TabularDatasetFactory
  - Transform and clean data: the clean_data() function inside *train.py* script is used again
  - Split data into train and test set: 80% train, 20% test
  - Concatenate features and target of training data in order to be correctly used in AutoMLConfig
  - Instantiate AutoMLConfig class with the parameters listed above
  - Submit AutoML run

After submitting the automl run to the experiment, the best run and the related metrics have been collected:
```
best_run_automl, best_model_automl = automl.get_output()
best_run_metrics_automl = best_run_automl.get_metrics()

print('Best run ID: ', best_run_automl.id)
print('Best run Accuracy: ', best_run_metrics_automl['Accuracy'])
print('Metrics: ', best_run_metrics_automl)
```
```
Best run ID: AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29
Best run Accuracy: 0.9190440060698029
Metrics: {'f1_score_macro': 0.7939297719210586, 'recall_score_micro': 0.9190440060698029, 'matthews_correlation': 0.5881065523806792, 'norm_macro_recall': 0.581082337069211, 'AUC_micro': 0.9808406538623611, 'precision_score_weighted': 0.9182078162705778, 'precision_score_macro': 0.7976971592815697, 'recall_score_weighted': 0.9190440060698029, 'AUC_weighted': 0.9481838936880631, 'recall_score_macro': 0.7905411685346055, 'balanced_accuracy': 0.7905411685346055, 'f1_score_weighted': 0.9185753244363074, 'f1_score_micro': 0.9190440060698029, 'weighted_accuracy': 0.9509449763133989, 'log_loss': 0.22503428584436114, 'average_precision_score_weighted': 0.9560620267940345, 'AUC_macro': 0.948183893688063, 'accuracy': 0.9190440060698029, 'average_precision_score_micro': 0.9815596471253963, 'average_precision_score_macro': 0.8271358405542905, 'precision_score_micro': 0.9190440060698029, 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29/confusion_matrix', 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29/accuracy_table'}
```

The **best model** was ***VotingEnsemble*** and it has been stored [here](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/outputs/best_model_automl.joblib):
  - ***Accuracy***: 0.9190440060698029

## Pipeline comparison
** Compare the two models and their performance. **
Performances (accuracy) obtained:
  - HyperDrive: 0.9072837632776934
  - AutoML: 0.9190440060698029

AutoML model performs slightly better than HyperDrive one.

Although both the approaches used the same preprocessing steps (clean_data() function), Hyperdrive train the model specified in the *train.py* script, the logistic regression, and tune the hyperparameters thanks to HyperDriveConfig class.
On the other hand, AutoML has the ability to train many algorithms in an easy/automatic way, and in this case used *voting ensemble* technique to combine predictions of different models in order to improve the performances

Anyway, AutoML pipeline took longer to complete than HyperDrive one, as shown below:
![Comparison](https://github.com/peppegili/1_Optimizing_an_ML_Pipeline_in_Azure/blob/master/img/comparison.png)

## Future work
** What are some areas of improvement for future experiments? Why might these improvements improve the model?**
