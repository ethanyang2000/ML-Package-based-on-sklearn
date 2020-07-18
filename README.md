# ML-Package-based-on-sklearn
It is a class called ModelCompare with a few public methods and members
##METHODS
###__init__(inp)
1.**params**
inp= inp is a dictionary
the keys are nick names of models while values are official names of them
and they are all provided as strings
eg:
inp={'rf':'RandomForestClassifier','svc':"SVC"}

###loadData(x,y)
1.**PARAMS**
x= the characters of the whole dataset
y= labels of the whole dataset
2.**FUCTION**
load dataset to the object
           
###fit()
1.**FUNCTION**
fit the input dataset with default params
in this fuction, the input dataset will be split into train and test parts
and they are storaged in the private members __x_train __x_test and so on

###predict(test)
1.**PARAMS**
test= the dataset for testing
2.**FUNCTION**
predict the dataset with all the models storaged in private member __models
the results are stored in a public member predict_data
it is a dictionary whose keys are nicknames of models while values are lists of predictions

###showAccurancy()
1.**FUNCTION**
show accurancy value of all models
IT USES MODELS WITH DEFAULT PARAMS (I.E THE MODELS TRAINED BY FIT METHOD)
the datasets are split in fit method
the results are stored in a private member __scores
it is a dict whose keys are nicknames while values are float


###showCrossValScore()
1.**FUNCTION**
show cross val score of all models
the models have best params which are trained by GridSearch method
I.E THIS METHOD CAN ONLY BE USED AFTER YOU USE GRIDSEARCH METHOD
results are stored in a private member __cross_val_scores
it is a dict whose keys are nicknames while values are lists of scores


###GridSearch()
1.**FUNCTION**
search best params for all models
best params are stored in a private member __best_params
it is a dict whose keys are nicknames and values are dicts
these dicts has structures as {'param_name':param}

best scores are stored as private dict __best_scores

###OfficialVoting(inp,test)
1.**PARAMS**
inp= a list of strings consisting of names
names can be official names or nicknames
test= dataset for testing
2.**FUNCTION**
hard voting classifier with input models
return a dataframe of predictions
models are trained by this voting model

###HardVoting(inp,test)
1.**PARAMS**
inp= a list of strings consisting of names and they can be nicknames or official names
test= dataset for testing
2.**FUNCTION**
use models trained in gridsearch for hard voting
RESULTS ARE RETURNDED IN A LIST
