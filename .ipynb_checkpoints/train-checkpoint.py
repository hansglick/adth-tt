from myfun import *

print("Downloading training set ...")
csv_url = "https://s3-eu-west-1.amazonaws.com/adthena-ds-test/trainSet.csv"
urllib.request.urlretrieve(csv_url, 'trainingset.csv')
df =pd.read_csv("trainingset.csv",header = None,names = ["request","category"])

print("Downloading test set ...")
csv_url = "https://s3-eu-west-1.amazonaws.com/adthena-ds-test/candidateTestSet.txt"
urllib.request.urlretrieve(csv_url, 'testset.txt')

print("Initializing dictionnaries ...")
dic_of_words = AutoVivification()
WordToRequest = {}
dic_of_requests = {}

print("Starting the loop ...")
for idrow,row in df.iterrows():
    print("\rProgression : ",round((idrow*100)/len(df)),"% of the dataset",end="")
    
    # Extract informations
    raw_request = row[0].lower()
    words = raw_request.split(" ")
    category = row[1]
    
    # Fill the dictionnary
    dic_of_requests[idrow] = {"request" : raw_request, "category" : row[1]}
    
    # Fill the dictionnaries
    for w in words:
        dic_of_words[w][category] = dic_of_words[w].get(category,0) + 1
        
        
        if w in WordToRequest:
            WordToRequest[w].append(idrow)
        else:
            WordToRequest[w] = [idrow]


print("Saving dictionnaries ...")
SavePickle("dic_of_words.p",dic_of_words)
SavePickle("WordToRequest.p",WordToRequest)
SavePickle("dic_of_requests.p",dic_of_requests)
df.reset_index(drop = False,inplace = True)
df.rename(index = str, columns = {"index":"idrequest"},inplace = True)
SavePickle("df.p",df)