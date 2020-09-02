import argparse
from myfun import *
import re
import string


# command line example : python predict -i testexample.txt -o results.txt -hyper 0.85


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

print("Build the translator to remove punctuations ...")
translator = str.maketrans('', '', string.punctuation)

def Run_Predictions_For_File(args):
    
    with open(args.input_file) as file:
        with open(args.output_file, "w") as fhandle:
            for idrow,l in enumerate(file):
                print("\rPredicted requests : ",idrow+1,end="")
                request = l.rstrip("\n").lower().translate(translator)
                request = re.sub(' +', ' ', request)
                predicted_category =  Predict_One_String_Request(request,
                                                                 dic_of_words,WordToRequest,PowerDicDis,dic_of_requests,
                                                                 Overallp,DefaultCategory,
                                                                 args.hyperparameter,[args.hyperparameter])
                toprint = request + "," + str(predicted_category) + "\n"
                fhandle.write(toprint)
    print("")
    print("Predictions are saved at ",args.output_file)

    return None



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='test arguments')
	parser.add_argument("-i", "--input_file", help="input filename", required=True)
	parser.add_argument("-o", "--output_file", help="output filename", required=True)
	parser.add_argument("-hyper", "--hyperparameter", help="hyperparameter 0.95 is suggested",type=float,default=0.95)
	args = parser.parse_args()
	Run_Predictions_For_File(args)


