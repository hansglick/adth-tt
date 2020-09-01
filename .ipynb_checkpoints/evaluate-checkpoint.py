import argparse
from myfun import *
import sys

# commande line : python evaluate -n 10000 -bh 0.65 -ah 0.65 0.75 0.85 0.95

print("Loading 'words to categories' dictionnary ...")
dic_of_words = LoadPickle("dic_of_words.p")

print("Loading 'words to request's' dictionnary ...")
WordToRequest = LoadPickle("WordToRequest.p")

print("Loading 'requests to words' dictionnary ...")
dic_of_requests = LoadPickle("dic_of_requests.p")

print("Loading training set dataframe ...")
df = LoadPickle("df.p")

print("Evaluating average prediction ...")
DefaultCategory = df.category.value_counts().index[0]
OverallDisDic = GetOverallCategoryDistribution(df)
Overallp =  DotProductFromDic(OverallDisDic)

print("Building 'words'discriminant' power dictionnary ...")
PowerDicDis = GetPowerDicDis(Overallp,dic_of_words,0.01)




def evaluate_model(args):

    Predictions = []
    current_results = {}
    some_ids = np.random.choice(np.arange(len(df)),replace = False, size = args.nobs)

    acc_by_categories = AutoVivification()
    
    for incrementation,IdRequest in enumerate(some_ids):

        one_prediction,truth_cat = PredictOneObs(IdRequest,0.01,
                                       df,dic_of_words,WordToRequest,PowerDicDis,dic_of_requests,
                                       Overallp,DefaultCategory,
                                       args.bottom_hyp,args.array_hyp)
        Predictions.append(one_prediction)

        

        for threshold,acc in one_prediction:
            current_results[threshold] = current_results.get(threshold,0) + acc
            acc_by_categories[truth_cat][threshold]["good"] = acc_by_categories[truth_cat][threshold].get("good",0) + acc
            acc_by_categories[truth_cat][threshold]["total"] = acc_by_categories[truth_cat][threshold].get("total",0) + 1



        elapsed_time = (incrementation +1) /len(some_ids)
        elapsed_time = int(round(elapsed_time,3)*100)
        elapsed_time = str(elapsed_time) + "%"

        accuracy = [(thr,current_results[thr]/(incrementation + 1)) for thr in current_results]
        toprinted = " | Accuracy (Hyperparameter value) : "
        for thr,acc in accuracy :
            temp =  str(round(acc,3)) +"(" + str(thr)+ ")" + "  "
            toprinted = toprinted + temp

        print( "\rTime : ", elapsed_time ,toprinted,end="")
        #print("\rTime : " + elapsed_time + toprinted + '\r', sep='', end ='', file = sys.stdout , flush = False)



    statsdf = pd.DataFrame.from_dict({(i,j): acc_by_categories[i][j] 
                            for i in acc_by_categories.keys() 
                            for j in acc_by_categories[i].keys()},
                           orient='index')
    statsdf["accuracy"] = statsdf.good / statsdf.total
    statsdf.to_csv("accuracy_by_category.csv")
    print("")
    print("Accuracy by category are available at accuracy_by_category.csv")




    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test arguments')
    parser.add_argument("-n", "--nobs", help="Nombre d'observations validation set", type = int, required=True)
    parser.add_argument("-bh", "--bottom_hyp", help="Bottom hyperparameter value", type = float, required=True)
    parser.add_argument("-ah", "--array_hyp", nargs='+', help="List of hyperparameter values to test",type = float,required=True)
    args = parser.parse_args()
    evaluate_model(args)


