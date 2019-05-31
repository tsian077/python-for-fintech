import datetime as dt 
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web 
#pandas_datareader.data用來抓取yahoo finance api


style.use('ggplot')

#start  = dt.datetime(2000,1,1)
#end = dt.datetime(2016,12,31)


#data frame  TSLA=特斯拉 從yahoo抓取資料
#df = web.DataReader('TSLA','yahoo',start,end)
#用來debug用，可以輸出資料 也可以用  df.tail(6)會得到最尾的6筆資料 df.head(6)得到最頭的
# print(df.head(6))
#df.to_csv('~/jimmy/python-for-finance/tsla.csv')

#讀取我們上面輸出的csv檔案,index_col=0時前面的索引數字會消失，parse_dates=true會給date_time_index
df = pd.read_csv('tsla.csv',index_col=0,parse_dates=True)
#print(df.head())

#Print出Open和High的資料
# print(df[['Open','High']].head())


#可以拿來畫圖
# df.plot()
# plt.show()

#新增一個新的column，取adj close的前一百個做平均
#df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()
#可以去掉nan的值
#df.dropna((inplace=Ture)

#adj close(adjusted closing price)為收盤價,
#將adj close數據計為10天一個，並做ohlc(open,high,low,close)折線圖
df_ohlc = df['Adj Close'].resample('10D').ohlc()
#volume為成交量,將十天的成交量加在一起
df_volume = df['Volume'].resample('10d').sum()

print(df_ohlc.head())

df_ohlc.reset_index(inplace=True)

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


#跟matlab的subplot一樣將圖畫在同個視窗
ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)


#ax1.plot(df.index, df['Adj Close'])
#ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index,df['Volumn'])

plt.show()


