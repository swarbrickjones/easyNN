import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import preprocessing
import math
from mLPClassifier import MLPClassifier
 
# Load training data
print 'Loading training data.'
data_train = np.loadtxt( 'HBdata/training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
 
# Pick a random seed for reproducible results. Choose wisely!
np.random.seed(42)
# Random number for training/validation splitting
r =np.random.rand(data_train.shape[0])
 
# Put Y(truth), X(data), W(weight), and I(index) into their own arrays
print 'Assigning data to numpy arrays.'
# First 90% are training

scaler = preprocessing.StandardScaler().fit(data_train[:,1:31])

X_train = scaler.transform(data_train[:,1:31][r<0.9])
y_train = data_train[:,32][r<0.9]
W_train = data_train[:,31][r<0.9]
# Lirst 10% are validation
X_valid = scaler.transform(data_train[:,1:31][r>=0.9])
y_valid = data_train[:,32][r>=0.9]
W_valid = data_train[:,31][r>=0.9]
 
# Train the GradientBoostingClassifier using our good features
print 'Training classifier (this may take some time!)'

#create classifier
num_features = X_train.shape[1]
clf = MLPClassifier(num_features, 2, n_epochs = 100, \
            n_hidden=100,learning_rate=0.01,  \
            L1_reg=0.00, L2_reg=0.000001,  \
            batch_size=20) \

clf.fit(X_train,y_train)

Yhat_train = clf.predict(X_train)
Yhat_valid = clf.predict(X_valid)

print(Yhat_train[0:100])

# Benchmark code//
#gbc = GBC(n_estimators=50, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1)
#gbc.fit(X_train,y_train)  
#prob_predict_train = gbc.predict_proba(X_train)[:,1]
#prob_predict_valid = gbc.predict_proba(X_valid)[:,1]
#pcut = np.percentile(prob_predict_train,85)
#Yhat_train = prob_predict_train > pcut 
#Yhat_valid = prob_predict_valid > pcut
# //Benchmark code

 
# To calculate the AMS data, first get the true positives and true negatives
# Scale the weights according to the r cutoff.
TruePositive_train = W_train*(y_train==1.0)*(1.0/0.9)
TrueNegative_train = W_train*(y_train==0.0)*(1.0/0.9)
TruePositive_valid = W_valid*(y_valid==1.0)*(1.0/0.1)
TrueNegative_valid = W_valid*(y_valid==0.0)*(1.0/0.1)
 
# s and b for the training 
s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut='
def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid)
 
# Now we load the testing data, storing the data (X) and index (I)
print 'Loading testing data'
data_test = np.loadtxt( 'HBdata/test.csv', delimiter=',', skiprows=1 )
X_test = data_test[:,1:31]
I_test = list(data_test[:,0])
 
# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
Predictions_test = gbc.predict_proba(X_test)[:,1]
# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 
# Now we get the CSV data, using the probability prediction in place of the ranking
print 'Organizing the prediction results'
resultlist = []
for x in range(len(I_test)):
    resultlist.append([int(I_test[x]), Predictions_test[x], 's'*(Label_test[x]==1.0)+'b'*(Label_test[x]==0.0)])
 
# Sort the result list by the probability prediction
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1]) 
 
# Loop over result list and replace probability prediction with integer ranking
for y in range(len(resultlist)):
    resultlist[y][1]=y+1
 
# Re-sort the result list according to the index
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])
 
# Write the result list data to a csv file
print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
fcsv = open('HBdata/NN_prediction_output.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline) 
fcsv.close()