import urllib.request
import pandas as pd
import numpy as np
import pickle
from itertools import combinations


def GrabData(csv_url="https://s3-eu-west-1.amazonaws.com/adthena-ds-test/trainSet.csv"):
    urllib.request.urlretrieve(csv_url, 'trainingset.csv')
    df =pd.read_csv("trainingset.csv",header = None,names = ["request","category"])
    return df


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
        
def SavePickle(filename,myobject):
    with open(filename, 'wb') as handle:
        pickle.dump(myobject, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def LoadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def ComputeStatsDic(df):

    dic_of_requests = {}
    dic_of_words = AutoVivification()
    WordToRequest = {}
    bigram = False

    for idrow,row in df.iterrows():
        print("\rProgression : ",round((idrow*100)/len(df)),"%",end="")

        # Extract informations
        words = row[0].split(" ")
        category = row[1]


        # Fill the dictionnary
        dic_of_requests[idrow] = words

        # Fill the dictionnary
        for w in words:
            dic_of_words[w][category] = dic_of_words[w].get(category,0) + 1

            if w in WordToRequest:
                WordToRequest[w].append(idrow)
            else:
                WordToRequest[w] = [idrow]




    return dic_of_words,WordToRequest,dic_of_requests


def Import_Data_And_Build_Basics_Stats(SaveBool=True):
    
    print("Load dataset from AWS")
    df = GrabData()
    
    print("Build Basic Dics")
    dic_of_words,WordToRequest,dic_of_requests = ComputeStatsDic(df)
    df.reset_index(drop = False,inplace = True)
    df.rename(index = str, columns = {"index":"idrequest"},inplace = True)

    print("Save data and Basic Dics")
    if SaveBool:
        SavePickle("dic_of_words.p",dic_of_words)
        SavePickle("WordToRequest.p",WordToRequest)
        SavePickle("df.p",df)
        SavePickle("dic_of_requests.p",dic_of_requests)
    
    return df,dic_of_words,WordToRequest,dic_of_requests


def Load_Data_And_Basic_Stats():
    
    try:
        dic_of_words = LoadPickle("dic_of_words.p")
        WordToRequest = LoadPickle("WordToRequest.p")
        df = LoadPickle("df.p")
        dic_of_requests = LoadPickle("dic_of_requests.p")
    except:
        print("Please check whether files exist")

    return dic_of_words,WordToRequest,df,dic_of_requests











def Build_Discriminant_Words_Power(df,dic_of_words,granularite):
    
    DefaultCategory = df.category.value_counts().index[0]
    OverallDisDic = GetOverallCategoryDistribution(df)
    Overallp =  DotProductFromDic(OverallDisDic)
    PowerDicDis = GetPowerDicDis(Overallp,dic_of_words,granularite)
    
    return DefaultCategory,OverallDisDic,Overallp,PowerDicDis









def GetInformationsFromIDRequest(IdRequest,df):
    SelectedRequest = df.request[IdRequest]
    SelectedCategory = df.category[IdRequest]
    SelectedWords = SelectedRequest.split(" ")
    return SelectedRequest, SelectedCategory, SelectedWords

def GetAverageP(n,granularite):
    
    p = np.arange(0,1+granularite,granularite)
    likelihood = np.power(p,n)
    cumsum = np.cumsum(likelihood) / likelihood.sum()
    idx_median = np.argmin(abs(cumsum - 0.5))
    solution = p[idx_median]
    
    return solution


def GetDistributionFromDic(mydic):
    
    NREQUETES = sum(mydic.values())
    d = {}
    for k,v in mydic.items():
        d[k] = v/NREQUETES
    
    return d,NREQUETES

def DistributionCategoryFromWord(myword,dic_of_words,granularite):
    mydic = dic_of_words[myword]
    DicDis,NREQUETES = GetDistributionFromDic(mydic)
    p_interest = GetAverageP(NREQUETES,granularite)
    d = {}
    for k,v in DicDis.items():
        d[k] = v*p_interest
    return d


def DotProductFromDic(mydic):
    s = np.array(list(mydic.values()))
    solution = np.dot(s,s)
    return solution


def GetDiscriminantPower(mydic,overallp):
    p = DotProductFromDic(mydic)
    power = p / overallp
    return power


def GetOverallCategoryDistribution(df):
    mydic = df.category.value_counts().to_dict()
    NR = len(df)
    d = {}
    for k,v in mydic.items():
        d[k] = v/NR
    return d

def GetPowerDicDis(Overallp,dic_of_words,granularite):
    a = {}
    for word in dic_of_words.keys():
        WordDisDic = DistributionCategoryFromWord(word,dic_of_words,granularite)
        a[word] = GetDiscriminantPower(WordDisDic,Overallp)
    return a

def GetRequestWeight(myrequest,PowerDicDis):
    
    splitted_words = myrequest.split(" ")
    solution = {}
    
    for word in splitted_words:
        if word in PowerDicDis:
            Power = PowerDicDis[word]
        else:
            Power = 1    
        solution[word] = Power
    
    return solution




def RetrieveRelevantRequestsFromDic(RequestDic,WordToRequest,df):
    
    L = []
    for word,powerdis in RequestDic.items():
        if word in WordToRequest:
            if powerdis > 0.0000001:
                idrequest = WordToRequest[word]
                L = L + idrequest
    L = np.array(L)
    L = np.unique(L)
    #REQUESTS = [(item,df.request[df.idrequest == item][0]) for item in L]
    #REQUESTS = [(item,df.request[item]) for item in L]
    REQUESTS = df[df.idrequest.isin(L)].to_dict('split')["data"]
    
    
    return REQUESTS

def RetrieveRelevantRequestsFromDic(RequestDic,WordToRequest,df,threshold):
    
    L = []
    retained_words = Retrieve_Relevant_Words_Only(RequestDic,threshold)
    
    for word in retained_words:
        if word in WordToRequest:
            idrequest = WordToRequest[word]
            L = L + idrequest
    L = np.array(L)
    L = np.unique(L)
    #REQUESTS = [(item,df.request[df.idrequest == item][0]) for item in L]
    #REQUESTS = [(item,df.request[item]) for item in L]
    REQUESTS = df[df.idrequest.isin(L)].to_dict('split')["data"]
    
    
    return REQUESTS








def GetPredictedCategory(REQUESTS,TargetWords,RequestDic,df,HYPERPARAMETER_inf,DefaultCategory,granularite):
    
    if len(REQUESTS)>0:
        RequestScoreDic = []

        for idrequest,r,cat in REQUESTS:
            #print(idrequest,r)
            RequestWords = r.split(" ")
            intersection = set(TargetWords).intersection(RequestWords)
            if len(intersection)>0:
                score = 0
                for w in intersection:
                    score = score + RequestDic[w]
            else:
                score = 0
            RequestScoreDic.append((r,score / sum(RequestDic.values()),df.category[idrequest]))
        
        RequestScoreDic.sort(key=lambda x : -x[1])
        RequestScoreDF = pd.DataFrame(RequestScoreDic,columns = ["request","score","category"])
        
        package = []
        for HYPERPARAMETER in np.arange(HYPERPARAMETER_inf,1,0.05):
            
            for HYP in np.arange(HYPERPARAMETER,0-granularite,-granularite):
                temp = RequestScoreDF[RequestScoreDF.score>=HYP]
                if len(temp) > 0:
                    break
            
            stats = temp.groupby('category')['score'].sum()
            solution = stats.idxmax()
            package.append((round(HYPERPARAMETER,3),solution))

    else:
        solution = DefaultCategory
        RequestScoreDF = pd.DataFrame()
        package = []
        for HYPERPARAMETER in np.arange(HYPERPARAMETER_inf,1,0.05):
            package.append((round(HYPERPARAMETER,3),solution))
    
    return package


def PredictOneObs(IdRequest,hypth,df,dic_of_words,WordToRequest,PowerDicDis,DefaultCategory,granularite,Overallp,threshold):
    
    SelectedRequest, SelectedCategory, SelectedWords = GetInformationsFromIDRequest(IdRequest,df)

    UpdateDic(IdRequest,SelectedRequest, SelectedCategory, SelectedWords,dic_of_words,WordToRequest,True)
    sauvegardePowerDicDis = SavePowerDicDis(SelectedWords,PowerDicDis)
    UpdatePowerDicDis(SelectedWords,dic_of_words,sauvegardePowerDicDis,granularite,Overallp,PowerDicDis,True)

    RequestDic = GetRequestWeight(SelectedRequest,PowerDicDis)
    REQUESTS = RetrieveRelevantRequestsFromDic(RequestDic,WordToRequest,df,threshold)
    TargetWords = list(RequestDic.keys())

    package = GetPredictedCategory(REQUESTS,TargetWords,RequestDic,df,hypth,DefaultCategory,granularite)

    UpdateDic(IdRequest,SelectedRequest, SelectedCategory, SelectedWords,dic_of_words,WordToRequest,False)
    UpdatePowerDicDis(SelectedWords,dic_of_words,sauvegardePowerDicDis,granularite,Overallp,PowerDicDis,False)
    
    accuracy_array = []
    for threshold,predicted_category in package:
        acc = predicted_category == SelectedCategory
        accuracy_array.append((threshold,acc))

    return accuracy_array,SelectedCategory









# Update dic_of_words
def UpdateDic(IdRequest,SelectedRequest, SelectedCategory, SelectedWords,dic_of_words,WordToRequest,movein):
    
    #dic_of_words_copy = copy.deepcopy(dic_of_words)
    #WordToRequest_copy = copy.deepcopy(WordToRequest)
    
    #dic_of_words_copy = dic_of_words.copy()
    #WordToRequest_copy = WordToRequest.copy()
    
    
    
    
    if movein:
        for word in SelectedWords:
            dic_of_words[word][SelectedCategory] = dic_of_words[word][SelectedCategory] - 1
            WordToRequest[word].remove(IdRequest)
            if dic_of_words[word][SelectedCategory] == 0:
                del dic_of_words[word][SelectedCategory]
    else:
        for word in SelectedWords:
            dic_of_words[word][SelectedCategory] = dic_of_words[word].get(SelectedCategory,0) + 1
            WordToRequest[word].append(IdRequest)
        
    return None

def SavePowerDicDis(SelectedWords,PowerDicDis):
    sauvegardePowerDicDis = {}
    for word in SelectedWords:
        sauvegardePowerDicDis[word] = PowerDicDis[word]
    return sauvegardePowerDicDis


def UpdatePowerDicDis(SelectedWords,dic_of_words,sauvegardePowerDicDis,granularite,Overallp,PowerDicDis,movein):

    if movein:
        for word in SelectedWords:
            WordDisDic = DistributionCategoryFromWord(word,dic_of_words,granularite)
            PowerDicDis[word] = GetDiscriminantPower(WordDisDic,Overallp)

    else:
        for word in SelectedWords:
            PowerDicDis[word] = sauvegardePowerDicDis[word]
    
    return None









def Evaluate_Accuracy(df,
                      dic_of_words,WordToRequest,PowerDicDis,
                      DefaultCategory,Overallp,threshold,
                      nobs=10000,granularite=0.01,hypth=0.65):

    # Empty variables
    some_ids = np.random.choice(np.arange(len(df)),replace = False, size = nobs)
    Predictions = []
    current_results = {}

    # Start
    for incrementation,IdRequest in enumerate(some_ids):

        one_prediction = PredictOneObs(IdRequest,hypth,df,dic_of_words,WordToRequest,PowerDicDis,DefaultCategory,granularite,Overallp,threshold)
        Predictions.append(one_prediction)


        for threshold,acc in one_prediction:
            current_results[threshold] = current_results.get(threshold,0) + acc


        elapsed_time = (incrementation +1) /len(some_ids)
        elapsed_time = int(round(elapsed_time,3)*100)
        elapsed_time = str(elapsed_time) + "%"

        accuracy = [(thr,current_results[thr]/(incrementation + 1)) for thr in current_results]
        toprinted = " | Acc : "
        for thr,acc in accuracy :
            temp =  str(round(acc,3)) + "  "
            toprinted = toprinted + temp

        print( "\rTested Requests : ",incrementation+1,"(",elapsed_time,")" ,toprinted, end="")

    return None






def Model_Evaluation(threshold,UsedPretrained=True):
    
    if UsedPretrained:
        print("Loading data and saved basic stats ...")
        dic_of_words,WordToRequest,df,dic_of_requests = Load_Data_And_Basic_Stats()
    else:
        print("Importing data and building basic stats ...")
        df,dic_of_words,WordToRequest,dic_of_requests = Import_Data_And_Build_Basics_Stats(SaveBool=True)

    print("Computing word's discriminant power ...")
    DefaultCategory,OverallDisDic,Overallp,PowerDicDis = Build_Discriminant_Words_Power(df,dic_of_words,0.01)

    print("Evaluating model's accuracy for different values for the hyperparameter ...")
    Evaluate_Accuracy(df,
                      dic_of_words,WordToRequest,PowerDicDis,
                      DefaultCategory,Overallp,threshold)
    return None












def Get_Predictions_From_Data_File(filename,firstnobs,UsedPretrained,granularite,HYPERPARAMETER,threshold):
    
    if UsedPretrained:
        print("Loading data and saved basic stats ...")
        dic_of_words,WordToRequest,df,dic_of_requests = Load_Data_And_Basic_Stats()
    else:
        print("Importing data and building basic stats ...")
        df,dic_of_words,WordToRequest,dic_of_requests = Import_Data_And_Build_Basics_Stats(SaveBool=True)

    print("Computing word's discriminant power ...")
    DefaultCategory,OverallDisDic,Overallp,PowerDicDis = Build_Discriminant_Words_Power(df,dic_of_words,0.01)

    print("Computing predictions ...")
    predicted_array = Compute_Predictions_From_File(filename,firstnobs,
                                                    PowerDicDis,WordToRequest,df,
                                                    DefaultCategory,granularite,
                                                    HYPERPARAMETER,threshold)
    
    print("Writing predictions ...")
    Write_Predictions(predicted_array,filename)
    
    return None


def Compute_Predictions_From_File(filename,firstnobs,
                                      PowerDicDis,WordToRequest,df,
                                      DefaultCategory,granularite,
                                      HYPERPARAMETER,threshold):

    predicted_array = []
    compteur = 0

    with open(filename) as f:
        for line in f:
            
            compteur = compteur + 1
            print("\rPredicted Requests : " , compteur,end="")

            myrequest = line.strip()
            temp_predicted = Compute_Prediction_From_A_Request(myrequest,
                                                               PowerDicDis,WordToRequest,df,
                                                               DefaultCategory,granularite,
                                                               HYPERPARAMETER,threshold)
            predicted_array.append(myrequest+","+str(temp_predicted))
            
            if compteur>firstnobs:
                break

    return predicted_array


def Write_Predictions(predicted_array,filename):
    
    savedfilename = filename + "_predictions.txt"
    
    with open(savedfilename, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(predicted_array))
    print("Predictions are available in the ",savedfilename," file!")
    
    return None

def Compute_Prediction_From_A_Request(SelectedRequest,
                                      PowerDicDis,WordToRequest,df,
                                      DefaultCategory,granularite,
                                      HYPERPARAMETER,threshold):
    
    RequestDic = GetRequestWeight(SelectedRequest,PowerDicDis)
    REQUESTS = RetrieveRelevantRequestsFromDic(RequestDic,WordToRequest,df,threshold)
    TargetWords = list(RequestDic.keys())
    predicted_category = Prediction_From_Request(REQUESTS,TargetWords,RequestDic,df,HYPERPARAMETER,DefaultCategory,granularite)
    
    return predicted_category


def Prediction_From_Request(REQUESTS,TargetWords,RequestDic,df,HYPERPARAMETER,DefaultCategory,granularite):
    
    if len(REQUESTS)>0:
        RequestScoreDic = []

        for idrequest,r,cat in REQUESTS:
            #print(idrequest,r)
            RequestWords = r.split(" ")
            intersection = set(TargetWords).intersection(RequestWords)
            if len(intersection)>0:
                score = 0
                for w in intersection:
                    score = score + RequestDic[w]
            else:
                score = 0
            RequestScoreDic.append((r,score / sum(RequestDic.values()),df.category[idrequest]))
        
        RequestScoreDic.sort(key=lambda x : -x[1])
        RequestScoreDF = pd.DataFrame(RequestScoreDic,columns = ["request","score","category"])
        

        
            
        for HYP in np.arange(HYPERPARAMETER,0-granularite,-granularite):
            temp = RequestScoreDF[RequestScoreDF.score>=HYP]
            if len(temp) > 0:
                break

        stats = temp.groupby('category')['score'].sum()
        solution = stats.idxmax()
        

    else:
        solution = DefaultCategory

    
    return solution







def Retrieve_Relevant_Words_Only(ff,threshold):

    words_array = list(ff.keys())
    scores_array = np.array(list(ff.values()))
    order_idx = np.argsort(scores_array)
    scores_array = scores_array[order_idx]
    cumsum_array = np.cumsum(scores_array) / scores_array.sum()
    retained_ids = list(order_idx[cumsum_array>threshold])
    retained_words = [w for idw,w in enumerate(words_array) if idw in retained_ids]

    return retained_words




















def Define_Partitions(words):

    L = []
    idx = list(range(len(words)))
    bigset = set(idx)

    for size in range(1,len(words)+1):
        temp = list(combinations(idx, size))
        negative_temp = [tuple(bigset - set(item)) for item in temp]
        pair = list(zip(temp,negative_temp))
        L = L + pair
    
    return L

def Add_Score_To_Partition(partitions,RequestDic_tuples):
    
    maximal_score = sum([item[1] for item in RequestDic_tuples])
    L = []
    
    for keep,remove in partitions:
        score = sum([RequestDic_tuples[item][1] for item in keep])
        L.append(((keep,remove),score,(score+0.00001)/(maximal_score+0.00001)))
    L.sort(key = lambda a : -a[1] )
    
    return L



def Get_Keep_Requests_From_Set(p,ranked_words,WordToRequest):
    translated_words = [ranked_words[item] for item in p]
    list_of_requests = [WordToRequest.get(w,[]) for w in translated_words]
    if len(list_of_requests)>0:
        solution = Get_Intersection(list_of_requests)
    else:
        solution = []
    return solution





def Get_Remove_Requests_From_Set(p,ranked_words,WordToRequest):
    translated_words = [ranked_words[item] for item in p]
    list_of_requests = [WordToRequest.get(w,[]) for w in translated_words]
    if len(list_of_requests)>0:
        solution = Get_Union(list_of_requests)
    else:
        solution = []
    return solution



def Build_Requested_Requests_Dic(partitions,ranked_words,WordToRequest):
    
    RR_Keep_Dic = {}
    RR_Remove_Dic = {}
    
    for (pkeep,premove),s1,s2 in partitions:
        RR_Keep_Dic[pkeep] = Get_Keep_Requests_From_Set(pkeep,ranked_words,WordToRequest)
        RR_Remove_Dic[premove] = Get_Remove_Requests_From_Set(premove,ranked_words,WordToRequest)
                
    return RR_Keep_Dic,RR_Remove_Dic


def Get_Intersection(biglist):
    biglist = [set(item) for item in biglist]
    solution = list(set.intersection(*biglist))
    return solution

def Get_Union(biglist):
    biglist = [set(item) for item in biglist]
    solution = list(set.union(*biglist))
    return solution



def Size_Rank_Partitions(partitions,RR_Keep_Dic,RR_Remove_Dic,hyp):
    
    L = []
    the_cumsum = 0
    
    for (keepr,remover),score,score_relatif in partitions:
        a = RR_Keep_Dic[keepr]
        b = RR_Remove_Dic[remover]
        satisfied_request = list(np.setdiff1d(a,b,True))
        the_cumsum = the_cumsum + len(satisfied_request)
                
        sol = {"partition" : (keepr,remover),
               "score" : score,
               "score_relatif" : score_relatif,
               "keep_size" : len(a),
               "remove_size":len(b),
               "nrequests":len(satisfied_request),
               "cum_requests" : the_cumsum,
               "requests":satisfied_request}
        
        L.append(sol)
        
        if the_cumsum>0 and score_relatif<hyp:
            break
    
    
    return L



def Retrieve_Best_Candidate_Requests(partitions,dic_of_requests,details,hyper_array):

    L = []
    for p in partitions:
        
        if len(p["requests"])>0:
            score = p["score"]
            requests_ids = p["requests"]
            score_relatif = p["score_relatif"]
            
            final = [(rid,dic_of_requests[rid]["request"],dic_of_requests[rid]["category"],score,score_relatif) for rid in requests_ids]
                    
            L = L + final
            
    nesting_dolls,sorted_hyper = Extract_Compart_From_Requests_List(L,hyper_array)
    final_result = []
    
    
    if details:
    
        for idset,set_of_requests in enumerate(nesting_dolls):
            dftoreturn = pd.DataFrame(set_of_requests,columns = ["idrequest","request","category","score","score_relatif"])
            predicted_category = dftoreturn.groupby('category')['score'].sum().idxmax()
            final_result.append((sorted_hyper[idset],predicted_category,dftoreturn))
    else:
        for idset,set_of_requests in enumerate(nesting_dolls):
            predicted_category = Return_Predicted_Category_From_Candidates(set_of_requests)
            final_result.append((sorted_hyper[idset],predicted_category))
    
    
    return final_result





def Return_Predicted_Category_From_Candidates(relevant_candidates_requests):
    
    candidates = {}
    for idrequest,request,category,score,score_relatif in relevant_candidates_requests:
        candidates[category] = candidates.get(category,0) + score
    candidates = [(k,v) for k,v in candidates.items()]
    candidates.sort(key = lambda a : -a[1])
    res = candidates[0][0]
    
    return res



def Get_Candidates_Smartly(partitions,ranked_words,hyp,WordToRequest):
    
    the_cumsum = 0
    L = []
    
    if hyp<1:
    
        for (pkeep,premove),score,score_relatif in partitions:
            if the_cumsum>0 and score_relatif<hyp:
                break
            else:
                keep_set = Get_Keep_Requests_From_Set(pkeep,ranked_words,WordToRequest)
                remove_set = Get_Remove_Requests_From_Set(premove,ranked_words,WordToRequest)
                satisfied_request = list(np.setdiff1d(keep_set,remove_set,True))
                the_cumsum = the_cumsum + len(satisfied_request)

                sol = {"partition" : (pkeep,premove),
                       "score" : score,
                       "score_relatif" : score_relatif,
                       "keep_size" : len(keep_set),
                       "remove_size":len(remove_set),
                       "nrequests":len(satisfied_request),
                       "cum_requests" : the_cumsum,
                       "requests":satisfied_request}
                L.append(sol)
                
    
    else:
        
        for (pkeep,premove),score,score_relatif in partitions:
            if the_cumsum>hyp:
                break
            else:
                keep_set = Get_Keep_Requests_From_Set(pkeep,ranked_words,WordToRequest)
                remove_set = Get_Remove_Requests_From_Set(premove,ranked_words,WordToRequest)
                satisfied_request = list(np.setdiff1d(keep_set,remove_set,True))
                the_cumsum = the_cumsum + len(satisfied_request)

                sol = {"partition" : (pkeep,premove),
                       "score" : score,
                       "score_relatif" : score_relatif,
                       "keep_size" : len(keep_set),
                       "remove_size":len(remove_set),
                       "nrequests":len(satisfied_request),
                       "cum_requests" : the_cumsum,
                       "requests":satisfied_request}
                L.append(sol)
        
            
    return L



def Extract_Compart_From_Requests_List(request_list,hyper_array):
    if hyper_array[0]<1:
        compar,sorted_hyper = Extract_Compart_From_Requests_List_p(request_list,hyper_array)
    else:
        compar,sorted_hyper = Extract_Compart_From_Requests_List_n(request_list,hyper_array)
    return compar,sorted_hyper



def Extract_Compart_From_Requests_List_p(request_list,hyper_array):
    
    hyper_array.sort()
    hyper_array = hyper_array[::-1]
    idrow_max = -1
    H = []
    SOLUTION = []
    max_score = request_list[0][4]
    
    hyper_array_corrected = []
    for hyp in hyper_array:
        if hyp<=max_score:
            hyper_array_corrected.append(hyp)
        else:
            hyper_array_corrected.append(max_score)
            
    #print(hyper_array_corrected)
    #print("")
    
    for hyp in hyper_array_corrected:

        for idrow,(idrequest,request,category,score,score_relatif) in enumerate(request_list):
            
            #print(hyp)
            #print(idrequest,request,category,score,score_relatif)
            #print(hyp,score_relatif)
            
            if (idrow+1) == len(request_list) or score_relatif<hyp:
                #print("cdt1, append + break")
                H.append((hyp,idrow))
                break
            else:
                #print("cdt2, on continue")
                idrow_max = idrow
            
    #print(H)
    for hyp,finalrow in H:
        SOLUTION.append(request_list[:finalrow+1])

    return SOLUTION,hyper_array



def Extract_Compart_From_Requests_List_n(request_list,hyper_array):
    
    hyper_array.sort()
    L = []
    
    score_max_row_dic = {}
    for idrow,(idrequest,request,category,score,score_relatif) in enumerate(request_list):
        score_max_row_dic[score_relatif] = idrow+1
    
    score_max_row_tuple = [(k,v) for k,v in score_max_row_dic.items()]
    score_max_row_tuple.sort(key = lambda a : a[1])
    
    iterrmaxrow = [item[1] for item in score_max_row_tuple]
    
    dmap = Mapping_Top_N(hyper_array,iterrmaxrow)
    dmap = [v for k,v in dmap.items()]
    dmap.sort()
    
    for finalrow in dmap:
        L.append(request_list[:finalrow])
    
    return L,hyper_array



def Mapping_Top_N(topn,rowlist):

    d = {}
    for n in topn:
        d[n] = topn[-1]
        for r in rowlist:
            if n<=r:
                d[n]=r
                break
            else:
                continue
    return d


def Inter_Flat_List(l):
    
    C = []
    for subl in l:
        L = []
        for item in subl:
            L = L + item
        C.append(L)
    
    return C


def Successive_Smashing_List(L):
    
    FINAL = []
    for idlist in range(len(L)):
        FINAL.append(L[:idlist+1])
    
    return FINAL


def ALL_PREDICTIONS_FROM_PARTITIONS(ranked_words,RequestDic_tuples,dic_of_requests,WordToRequest,bottom_hyp,array_hyp,DefaultCategory,with_details=False):
    if len(ranked_words)>0:
        partitions = Define_Partitions(ranked_words)
        partitions = Add_Score_To_Partition(partitions,RequestDic_tuples)
        partitions = Get_Candidates_Smartly(partitions,ranked_words,bottom_hyp,WordToRequest)
        PREDICTIONS = Retrieve_Best_Candidate_Requests(partitions,dic_of_requests,with_details,array_hyp)
    else:
        PREDICTIONS = [(item,DefaultCategory) for item in array_hyp]
    return PREDICTIONS


def Clean_Ranked_Words(ranked_words,WordToRequest):
    
    ranked_words_II = []
    
    for w in ranked_words:
        cdta = w in WordToRequest
        if cdta:
            cdtb = WordToRequest[w] != []
        else:
            cdtb = False
        if cdta+cdtb == 2:
            ranked_words_II.append(w)
    
    return ranked_words_II




















def Predict_One_String_Request(SelectedRequest,
                               dic_of_words,WordToRequest,PowerDicDis,dic_of_requests,
                               Overallp,DefaultCategory,
                               bottom_hyp,array_hyp):

    RequestDic = GetRequestWeight(SelectedRequest,PowerDicDis)
    RequestDic_tuples = [(k,v) for k,v in RequestDic.items()]
    RequestDic_tuples.sort(key=lambda tup: -tup[1])
    ranked_words = [item[0] for item in RequestDic_tuples if item[1]>0.0001]
    ranked_words = Clean_Ranked_Words(ranked_words,WordToRequest)

    PREDICTIONS = ALL_PREDICTIONS_FROM_PARTITIONS(ranked_words,
                                                  RequestDic_tuples,dic_of_requests,WordToRequest,
                                                  bottom_hyp,array_hyp,DefaultCategory,with_details=False)


    return PREDICTIONS[0][1]


def PredictOneObs(IdRequest,granularite,
                  df,dic_of_words,WordToRequest,PowerDicDis,dic_of_requests,
                  Overallp,DefaultCategory,
                  bottom_hyp,array_hyp):
    
    #print(IdRequest)

    SelectedRequest, SelectedCategory, SelectedWords = GetInformationsFromIDRequest(IdRequest,df)
    UpdateDic(IdRequest,SelectedRequest, SelectedCategory, SelectedWords,dic_of_words,WordToRequest,True)
    sauvegardePowerDicDis = SavePowerDicDis(SelectedWords,PowerDicDis)
    UpdatePowerDicDis(SelectedWords,dic_of_words,sauvegardePowerDicDis,granularite,Overallp,PowerDicDis,True)

    RequestDic = GetRequestWeight(SelectedRequest,PowerDicDis)
    RequestDic_tuples = [(k,v) for k,v in RequestDic.items()]
    RequestDic_tuples.sort(key=lambda tup: -tup[1])
    ranked_words = [item[0] for item in RequestDic_tuples if item[1]>0.0001]
    ranked_words = Clean_Ranked_Words(ranked_words,WordToRequest)

    PREDICTIONS = ALL_PREDICTIONS_FROM_PARTITIONS(ranked_words,
                                                  RequestDic_tuples,dic_of_requests,WordToRequest,
                                                  bottom_hyp,array_hyp,DefaultCategory,with_details=False)

    UpdateDic(IdRequest,SelectedRequest, SelectedCategory, SelectedWords,dic_of_words,WordToRequest,False)
    UpdatePowerDicDis(SelectedWords,dic_of_words,sauvegardePowerDicDis,granularite,Overallp,PowerDicDis,False)

    accuracy_array = []
    for threshold,predicted_category in PREDICTIONS:
        acc = predicted_category == SelectedCategory
        accuracy_array.append((threshold,acc))

    return accuracy_array,SelectedCategory