### Question 1, Selected  model

Dealing with very sparse data and asymmetric distributions of the features we chose a more naïve model than the parametric classical ones. We opted for a version of KNN. We first computed the discriminant powers of all the words in the vocabulary. Discriminant power of a word is a measure supposed to define the importance of a word when it comes to predict the category. Then we say that the similarity used to find the nearest queries neighbors between two queries is the sum of the discriminant powers of the words shared by the two queries. Finally, we use a specific strategy to reduce the cost of searching the nearest neighbors of a query target. Once the neighboring queries are retrieved, the weighted distribution of the category variable among them is computed. The mode of the distribution is chosen as the prediction. [More details about the algorithm we developped](https://github.com/hansglick/adth-tt/blob/master/docs/Model_details.pdf)

***

### Question 2, Pre-processing steps

Almost no pre-processing was done except making sure that :
 * Every character was in lower case
 * No extra blanks in the queries

***

### Question 3, Evaluation of the performance

Given the large number of classes we faced (~ 1400), it seemed wise to us to choose accuracy as the measure of performance of the model. We need to set the quality of neighboring queries we want to make the prediction. This quality is represented by the unique hyperparameter of the model. It turned out that h=0.95 give the best results. The strategy to evaluate the model is pretty simple. For each query target :
 * We correct the dictionnary that contains the discriminant powers of the words (because it was build on the entire training set)
 * We search among all the queries of the training set (except the query target), the nearest queries according to the chosen similarity
 * We compute the prediction
 * We update the accuracy
 * We update back the dictionnary that contains the discriminant powers of the words

***

### Question 4, Runtime complexity

#### Training

We do not have to train the model, strictly speaking, training does not exist for KNN. However, we need to compute a lot of statistics saved in dictionnaries. We need to build 4 dictionanries :

 * **Word to categories** : for one word, the distribution of the category variable
 * **ID query to words** : for one id query, the words in it and the category of the query
 * **Word to queries** : for one word, the queries that contains this word
 * **Discriminant powers** : for one word, its discriminant power

The first three dictionnaries can be build in one loop over the queries. Then, we need to loop over the words to estimate the probability density functions p(c|wi). Then, we need a second run on the words to compute the discriminant power for each word. Unfortunately, we cannot group these two loops in one. As a consequence we can say that the time complexity of the algorithm during the training part is:

> O(n1 + 2 n2)

 * *n1* : number of queries
 * *n2* : number of words


The model is based on 4 python dictionnaries. The memory occupied in RAM of these 4 dictionaries is around 28 megabytes.

***

#### Test time

At test time, the most expensive task is the extraction of the top n nearest queries. The naïve approach would suggest to, first, compute the similarity for each query that contains at least one word shared with the query target. This approach gave us around 3,000 candidates queries for a query target. However, as explained in the [slides](https://github.com/hansglick/adth-tt/blob/master/docs/Model_details.pdf), we opted for a better strategy : we review queries of decreasing quality until a certain predefined quality threshold is reached (the hyperparameter value). This strategy allow us to reduce drastically the runtime complexity since we want to make our predictions only on the very best quality (high similarity) neighboring queries. On average, with a hyperparameter value set to 0.95, we retrieve only 10 nearest queries
for a query target. We went from 0.25s (naïve approach) to 0.005s (best queries first) on average. We need to mention that our strategy is **not robust against queries with a lot of words**. Indeed, our first step is to rank all the combinations of query target's set of words.

***




### Question 5, Model weaknesses


The obvious weaknesses of the model are the following :

#### **1.** The algorithm considers the query as a bag of words

Which might not be too bad since the queries are somewhat short. However we can imagine that the query *boston globe* would be misclassified since the words *boston* and *globe* would be associated with other categories than newspaper. One of the solution would be to see a query as a bag of words and **a bag of bigrams**.

#### **2.** The algorithm considers every pair of non unique words as two different words

That might cause some problems especially in the following cases :
 * When entities do not have **permanent orthography**. Which is generally the case for foreign words used to represent people or places. To deal with this problem, we can imagine a way to reconcile words under specific circumstances based on levenshtein distance, the length of the words and their frequency.
 * When a query contains an **unknown or infrequent word**. To deal with this kind of problem, we might want to use word embeddings. Let's say the search engine is used so far by a population of a specific country in which the *pepsi* brand does not exist. Then, let's say pepsi has just been authorized to sell products in this country. At the beginning, the algorithm would treat *pepsi* as an item with no information which is a shame. Particlularly if the word *coca-cola* is frequent enough to let the algorithm make good prediction. Indeed we know as people that pepsi and coca-cola are different words used to represent the same kind of product. Moreover, this representation is accessible through pre-trained word embeddings. It means that, one might imagine a more powerful similarity between two queries that would take into account the embeddings of the queries's words. As a consequence, *bottle of pepsi* and *bottle of coca-cola* would be more similar than *bottle of pepsi* and *plastic bottle* even though *pepsi* would be an unknown word.


#### **3.** The algorithm considers a pair of the same word as similar

Which is what we want in most cases. However, we know that sometimes the same word can have different meanings given the context. For example, the word *bat* in the two following queries : *chicago yankees bat* ; *bat seen in chicago*. One way to adress this sort of problem would be to use **transformers**. As we know, transformer is a deep learning model that transform each word embedding of a sentence, according to the embeddings of the surrounded words. In other words, transformer sort of update initial word embeddings given the context (and the position of the word) which is exactly what we need in order to adress the word disambiguation problem.

