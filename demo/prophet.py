# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:49:17 2019

@author: dbt
"""
# Python
import pandas as pd
import numpy as np
from fbprophet import Prophet
import sys
import os
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot

#orig_out = sys.stdout
#sys.stdout = open(os.devnull, 'w')
 

#filename = sys.argv[1]  #从cmd读取数据
filename='data'#从指定文件读取数据


df = pd.read_csv(filename+'.txt')
#df['y'] = np.log(df['y'])  #为什么要log处理？都要有吗？

#df['cap'] = 1#log预测才用
#df['floor'] = -1#log预测才用

df.head()


#设置跟随性： changepoint_prior_scale=0.05 值越大，拟合的跟随性越好，可能会过拟合
#设置置信区间：interval_width=0.8（默认值）,值越小，上下线的带宽越小。
#指定预测类型： growth='linear'或growth = "logistic" ，默认应该是linear。
#马尔科夫蒙特卡洛取样（MCMC）： mcmc_samples=0,会计算很慢。距离意义不清楚
#设置寻找突变点的比例：changepoint_range=0.9 默认从数据的前90%中寻找异常数据。预测这个正弦曲线，如果不设置changepoint_range=1，预测的结果是不对的，不知道为什么。

m = Prophet(changepoint_prior_scale=0.9,interval_width=0.9,growth='linear',changepoint_range=1)          
m.fit(df);

#periods 周期，一般是根据实际意义确定，重点：后续预测的长度是一个周期的长度。
#freq 我见的有‘MS‘、H、M ，预测sin，要设置H ，个人理解数据如果变化很快，要用H
future = m.make_future_dataframe(periods=120, freq='H') #freq=‘MS‘或者H  来设置

future['cap'] = 1 #log预测才用？linear也可以加上。
future['floor'] = -1#log预测才用？linear也可以加上。

#画图
future.tail()

forecast = m.predict(future)
forecast.tail()
fig=m.plot(forecast)
plt.savefig('./out/'+filename+'_1.jpg',dpi=500)
m.plot_components(forecast)
plt.savefig('./out/'+filename+'_2.jpg',dpi=500)
#print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])  #打印到console
 

savename='./out/'+filename+"_out.txt"
 
 
#forecast.to_csv(savename, sep='\t',index=False)   #保留panda.dataframe 的全部列数据

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(savename, sep='\t',index=False) #保留panda.dataframe 的指定列的数据

x = forecast['ds']
y = forecast['yhat']
y1 = forecast['yhat_lower']
y2 = forecast['yhat_upper']
plt.plot(x,y)
plt.savefig('./out/'+filename+'_3.jpg',dpi=500)
plt.plot(x,y1)
plt.savefig('./out/'+filename+'_4.jpg',dpi=500)
plt.plot(x,y2)
plt.savefig('./out/'+filename+'_5.jpg',dpi=500)
#plt.show()

#把检测到的突变点，用红色线表示在图上。
a = add_changepoints_to_plot(fig.gca(), m, forecast)

