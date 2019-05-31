from collections import Counter
import numpy as np 
import pandas as pd 
import pickle 
from sklearn import svm,cross_validation,neighbors #導入SVM,交叉驗證,
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from matplotlib import pyplot as plt

def process_data_for_labels(ticker):
    #7天之內股價起伏
    hm_days = 7
    df = pd.read_csv('D:/MachineLearing/keras/python-for-finance/sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)

    for i in range(1, hm_days+1):
        #未來的股價-現在的股價/現在的股價
        #shift左移(csv中是上移)一個
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        
    #將nan的值替代成0
    #print(df['{}_{}d'.format(ticker,i)])
    df.fillna(0, inplace=True)
    return tickers,df


#process_data_for_labels('XOM_HL_pct_diff')

#*args 可以傳入可變化的參數，
def buy_sell_hold(*args):
    cols = [c for c in args]
    #用來檢測這間公司是不是漲了或跌了百分之二
    requirement = 0.025
    for col in cols:
        if col >  requirement:
            return 1
        if col < -requirement:
            return -1
    return 0
         

def extract_featuresets(ticker):
    tickers,df = process_data_for_labels(ticker)
    
    df['{}_target'.format(ticker)] =list(map(buy_sell_hold,
                                         df['{}_1d'.format(ticker)],
                                         df['{}_2d'.format(ticker)],
                                         df['{}_3d'.format(ticker)],
                                         df['{}_4d'.format(ticker)],
                                         df['{}_5d'.format(ticker)],
                                         df['{}_6d'.format(ticker)],
                                         df['{}_7d'.format(ticker)],
                                         ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0,inplace=True)

    #取代掉如果0突然變成有數字，或是有數字變成0 那就是有問題的用nan取代
    df =df.replace([np.inf,-np.inf],np.nan)
    #再把nan去掉
    df.dropna(inplace=True)
    
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)

    x = df_vals.values #每個公司的百分比變化
    y = df['{}_target'.format(ticker)].values #y是1或-1或0  0為hold 1為買 -1為賣

    return x,y, df 


#extract_featuresets('XOM_HL_pct_diff')

X,y ,df = extract_featuresets('XOM')
print(X[1])



def do_ml(ticker):
    #做knnclassifier
    x,y ,df = extract_featuresets(ticker)
    print('x',x)
    print('y',y)
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.25)

    #做knnclassifier
    clf = neighbors.KNeighborsClassifier()

    #VotingClassifier選出最好的classifier
    #clf = VotingClassifier([('lsvc',svm.LinearSVC()),
     #                       ('knn',neighbors.KNeighborsClassifier()),
      #                      ('rfor',RandomForestClassifier())])


    #fit()=train()會訓練我們的資料 
    clf.fit(x_train,y_train)
    confidence = clf.score(x_test,y_test)
    print('Accuracy:',confidence)
    predictions = clf.predict(x_test)
    print('Predicted spread:',Counter(predictions))

    return confidence

#do_ml('AVY_daily_pct_chng')




def svm_sgd(X, Y):

    w = np.zeros(len(X))
    eta = 1
    epochs = 100000


    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + eta * (-2  *(1/epoch)* w)

    return w

#w = svm_sgd(X,y)
#print(w)

def svm_sgd_plot(X, Y):

    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    errors = []


    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
                error = 1
            else:
                w = w + eta * (-2  *(1/epoch)* w)
        errors.append(error)

    plt.plot(errors, '|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

#svm_sgd_plot(X,y)