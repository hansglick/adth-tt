# Overview

#### **Goal and model** 
The objective is to correctly predict the category to which a query belongs. A query is a sequence of words. The training set contains about 70,000 words. Queries can belong to one of 1500 possible categories. In view of the sparsity of the data and the high number of classes, a KNN type model was chosen. **An accuracy higher than 58% was obtained**

 * For more details about the model I use, **go there**
 * For the answers to the specific answers you ask, **go there**
 * The following will give you instructions for using the model


***

# Application


#### **Clone the repository and set the environment**
Tou should replicate the environment under which the application was developed. To do this, please use conda. With the following commands, you will clone and activate the needed environment :

```
git clone https://github.com/hansglick/adth-tt.git
conda env create -f environment.yml
conda activate adthena
```

***

#### **Training**

It is not strictly speaking about training since it is a KNN type model. However, the model requires several statistics saved in dictionaries. The commands below allow to download the data and to build these statistics. However, be aware that this operation can take time. Moreover, it is not necessary to do it, the repository already contains already the needed files.

```
(adthena) python training.py
```

This command will download and build the following files : 

 * *trainingSet.csv* , training dataset
 * *testset.txt* , test dataset
 * *WordToRequest.p* , a pickle file that contains a dictionnary used by the model
 * *dic_of_words.p* , a pickle file that contains a dictionnary used by the model
 * *dic_of_requests.p* , a pickle file that contains a dictionnary used by the model
 * *df.p* , a pickle file that contains a pandas dataframe used by the model

***

#### **Evaluation**

For each target query to predict, the model will use the entire training set except the target query. In short, in order to predict the category of a target query, the model looks for the closest candidate queries in the training set. Then, it votes these candidates. The category with the most votes becomes the prediction. In order to evaluate the model, the accuracy was chosen. To start the evaluation of the model, use the following command. It will also write a file that contains the accuracy for each category and each tested hyperparameter value. Again, to get more details about the model, **PLEASE GO THERE**

```
(adthena) python evaluate.py -n 10000 -bh 0.65 -ah 0.65 0.75 0.85 0.95
```
 * **n** : Number of observations in the training set to use to evaluate the model. Although in theory the evaluation should be run over the entire training set, keep in mind that a prediction can take time
 * **bh** : Hyperparameter of the model
 * **ah** : List of values for the hyperparameter

If you want to explore the mistakes the model does, you can use [this notebook](https://github.com/hansglick/adth-tt/blob/master/Exploring_mistakes.ipynb)

***

#### **Prediction**

In order to use the model to predict the categories of queries contained in a text file, run the following command : 

```
(adthena) python predict.py -i testset.txt -o predictions.txt -hyper 0.95
```
 * **i** : Name of the input file, the file that contains the queries for which we want to predict the category
 * **o** : Name of the output file, it will contains the input queries and the predicted categories
 * **hyper** : Value of the hyperparameter, please use 0.95 since it gives good performance

