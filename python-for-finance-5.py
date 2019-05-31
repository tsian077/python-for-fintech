import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

from collections import Counter
from sklearn import svm,cross_validation,neighbors #導入SVM,交叉驗證,
from sklearn.ensemble import VotingClassifier,RandomForestClassifier

style.use('ggplot')

def save_sp500_tickers():
    #透過requests傳回我們的網址
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    #透過BeautifulSoip解析出我們的網站，lxml是解析器(Parser)速度較快
    soup = bs.BeautifulSoup(resp.text, 'lxml')

    #我們要找出每間公司的名稱，透過找出table，class是wikitable sortable
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        mapping = str.maketrans(".","-")
        ticker = ticker.translate(mapping)
        tickers.append(ticker)
    #tickers保存了SP500的每個公司的名稱
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers



#save_sp500_tickers()


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    #讀取從2015到現在2018的股票價格
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime.now()

    #透過tickers使用DataReader去yahoo抓取每個公司的股價，從2013到2018
    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


#get_data_from_yahoo()

def compile_data():
    #讀取sp500tickers，取得我們的tickers(公司名稱)
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    for count,ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            if not df.empty:
                #將date設為index
                df.set_index('Date', inplace=True)
            #將stock_dfs中每一個Adj Close(調整後的收盤價)換成ticker(公司名稱)
            df.rename(columns={'Adj Close':ticker}, inplace=True)
            #刪除Open,High,Low,Low,Close,Volumn的列(axis=1)，默認為行
            df.drop(['Open','High','Low','Close','Volume'],axis=1,inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df)
            if count %10 ==0:
                print(count)
            #print(main_df.head())
        except:
            print('Connot obtain data for')
    main_df.to_csv('sp500_joined_closes.csv')

#compile_data()


def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #corr 用作計算相關性，計算列的相關性
    #每天與其他公司的相關性，
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


#visualize_data()



def process_data_for_labels(ticker):
    #7天之內股價起伏
    hm_days = 7
    df = pd.read_csv('D:/MachineLearing/keras/python-for-finance/sp500_joined_closes.csv',index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    #從第一天到第二天的起伏到第一天到第七天的起伏
    for i in range(1, hm_days+1):
        #未來的股價-現在的股價/現在的股價
        #shift左移(csv中是上移)
        df['{}_{}d'.format(ticker,i)] = ((df[ticker].shift(-i) - df[ticker]) / df[ticker])
        
        
        
    #將nan的值替代成0
    #print(df['{}_{}d'.format(ticker,i)])
    df.fillna(0, inplace=True)
    return tickers,df


#process_data_for_labels('XOM')

#*args 可以傳入可變化的參數，
def buy_sell_hold(*args):
    cols = [c for c in args]
    #用來檢測這間公司是不是漲了或跌了百分之二
    requirement = 0.02
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
    #print(df['{}_1d'.format(ticker)])
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    #print('Data spread:',Counter(str_vals))

    df.fillna(0,inplace=True)

    #取代掉如果0突然變成有數字，或是有數字變成0 那就是有問題的用nan取代
    df =df.replace([np.inf,-np.inf],np.nan)
    #再把nan去掉
    df.dropna(inplace=True)
    
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)

    #x = (df_vals.values * 10).astype(int) #每個公司的百分比變化
    x = df_vals.values
    y = df['{}_target'.format(ticker)].values #y是1或-1或0  0為hold 1為買 -1為賣

    return x,y, df 

#extract_featuresets('XOM')


def do_ml(ticker):
    #做knnclassifier
    x,y ,df = extract_featuresets(ticker)
   
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

#do_ml('XOM')



X,y ,df = extract_featuresets('XOM')




print(X.shape)

print(y.shape)

def svm_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + eta * (-2  *(1/epoch)* w)

    return w

#w=svm_sgd(X,y)
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

for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples

plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V,scale=1, color='blue')

plt.show()
