#Haris Qureshi
#1001241073
#April 28,2020

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


#read from the csv file and return a Pandas DataFrame.
nba = pd.read_csv('NBAstats.csv')

# print the column names
original_headers = list(nba.columns.values)
#print(original_headers)

#print the first three rows.
#print(nba[0:3])

# "Position (pos)" is the class attribute we are predicting. 
class_column = 'Pos'
number = input("enter the number of features you want")
step = int(number)
oof = number+'.txt'
with open(oof) as f:
#The dataset contains attributes such as player name and team name.
#We know that they are not useful for classification and thus do not
#include them as features.
    content = f.readlines()
    content = [x.strip() for x in content]

ft = {1:'Age',2:'G', 3:'GS', 4:'MP', 5:'FG', 6:'FGA', 7:'FG%', 8:'3P', 9:'3PA',10:'3P%',11:'2P', 12:'2PA', 13:'2P%', 14:'eFG%', 15:'FT', 16:'FTA', 17:'FT%',18:'ORB', 19:'DRB',20:'TRB', 21:'AST', 22:'STL', 23:'BLK', 24:'TOV', 25:'PF', 26:'PS/G'}

feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
    '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
    'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']

j =0
best =0
k =1
bestl=[]
for i in content[::step]:
    j+=step
    feature_columns.clear()
    k =1
    while k < j and k !=27:
        feature_columns.append(ft[k])
        k+=1
    
    '''
    feature_columns = [ 'FGA','FG%',  '3PA', \
         '2PA',  'FT', 'FTA', 'ORB', 'DRB', \
        'TRB', 'AST', 'STL', 'BLK', 'TOV']
    '''
    '''
    0.598
    feature_columns = [ 'FGA','FG%',  '3PA', \
         '2PA',  'FT', 'FTA', 'ORB', 'DRB', \
        'TRB', 'AST', 'STL', 'BLK', 'TOV']
    0.594
    feature_columns = [ 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
        '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
        'TRB', 'AST', 'STL', 'BLK',  'PF', 'PS/G']

    feature_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', \
        '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', \
        'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
    '''
    #Pandas DataFrame allows you to select columns.
    #We use column selection to split the data into features and class.
    nba_feature = nba[feature_columns]
    nba_class = nba[class_column]

    #print(nba_feature[0:3])
    #print(list(nba_class[0:3]))

    train_feature, test_feature, train_class, test_class = train_test_split(nba_feature, nba_class, stratify=nba_class,train_size=0.75, test_size=0.25)

    training_accuracy = []
    test_accuracy = []

    #Linear SVM

    LSVM= LinearSVC(dual = False ,random_state=0).fit(train_feature,train_class)
    prediction = LSVM.predict(test_feature)

    #print("\nTest set predictions:\n{}".format(prediction))
    print("Test set accuracy: {:.2f}".format(LSVM.score(test_feature,test_class)))


    print("\n")
    #print("Confusion matrix:")
    #print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

    train_class_df = pd.DataFrame(train_class,columns=[class_column])
    train_data_df = pd.merge(train_class_df, train_feature, left_index=True, right_index=True)
    train_data_df.to_csv('train_data.csv', index=False)

    temp_df = pd.DataFrame(test_class,columns=[class_column])
    temp_df['Predicted Pos']=pd.Series(prediction, index=temp_df.index)
    test_data_df = pd.merge(temp_df, test_feature, left_index=True, right_index=True)
    test_data_df.to_csv('test_data.csv', index=False)

    crossvalScores = cross_val_score(LSVM,nba_feature, nba_class, cv=10)
    print(feature_columns)
    print("\n")
    print("Cross-validation scores: {}".format(crossvalScores))
    print("\n")
    print("Average cross-validation score: {:.3f}".format(crossvalScores.mean()))
    print("\n")
    if crossvalScores.mean() > best:
        bestl = feature_columns.copy()
        best=crossvalScores.mean()
    
print("\n")
print(bestl)
print("\n")
print("the best: {:.3f}".format(best))
