import pandas as pd
import numpy as np

first = pd.read_excel("2009.xlsx") 
list(first.columns.values)
second = pd.read_excel("2010.xlsx")
third = pd.read_excel("2011.xlsx")
f = first.iloc[:,[3,4,5]]
s = second.iloc[:,[3,4,5]]
t = third.iloc[:,[3,4,5]]
f['sales'] = f['Quantity']*f['Price']
s['sales'] = s['Quantity']*s['Price']
t['sales'] = t['Quantity']*t['Price']
f.drop(['Price','Quantity'], axis=1, inplace = True)
s.drop(['Price','Quantity'], axis=1, inplace = True)
t.drop(['Price','Quantity'], axis=1, inplace = True)

f['month'] = f['InvoiceDate'].dt.month 
s['month'] = s['InvoiceDate'].dt.month
t['month'] = t['InvoiceDate'].dt.month

f = f.groupby('month').sales.sum().reset_index()
s = s.groupby('month').sales.sum().reset_index()
t = t.groupby('month').sales.sum().reset_index()

f['month'] = ['Dec']
s['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
t['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
online = pd.concat([f,s,t])

# to change the index value in pandas data frame 
online = online.set_index(np.arange(1,26))

month_dummies = pd.DataFrame(pd.get_dummies(online['month']))
online = pd.concat([online,month_dummies],axis = 1)

online["t"] = np.arange(1,26)

online["t_squared"] = online["t"]*online["t"]

online["log_sales"] = np.log(online["sales"])

online.sales.plot()

Train = online.head(19)
Test = online.tail(6)

import statsmodels.formula.api as smf 

linear_model = smf.ols('sales~t',data=Train).fit()

pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))

rmse_linear = np.sqrt(np.mean((np.array(Test['sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


############################################################

online1 = online.head(24)

Train1 = online1.head(18)
Test1 = online1.tail(6)

linear_model = smf.ols('sales~t',data=Train1).fit()

pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test1['t'])))

rmse_linear = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train1).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test1['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('sales~t+t_squared',data=Train1).fit()
pred_Quad = pd.Series(Quad.predict(Test1[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=Train1).fit()
pred_add_sea = pd.Series(add_sea.predict(Test1[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=Train1).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test1[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data = Train1).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test1))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data = Train1).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test1))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test1['sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data1 = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse1=pd.DataFrame(data1)
table_rmse1
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 

predict_data = pd.read_excel("predict_online.xlsx")

predict_data["t_squared"] = predict_data["t"]*predict_data["t"]
predict_data['month'] = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

month_dummies = pd.DataFrame(pd.get_dummies(predict_data['month']))
predict_data = pd.concat([predict_data,month_dummies],axis = 1)


model_full = smf.ols('sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=online1).fit()

pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new

predict_data["forecasted_sales"] = pd.Series(pred_new)
###############################
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA

model_full = smf.ols('sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sept+Oct+Nov',data=online1).fit()
pred_full = model_full.predict(online1)
online1["pred_full"] = pred_full
residuals = pd.DataFrame(np.array(online1["sales"]-np.array(pred_full)))

tsa_plots.plot_acf(residuals,lags=12)

model= ARIMA(residuals, order= (1,0,0)).fit(disp=0)

ARresiduals = pd.DataFrame(model.resid)

tsa_plots.plot_acf(ARresiduals,lags=12)

forecast = model.forecast(steps=12)[0]
predict_data["forecasted_sales"] = pd.Series(pred_new)
predict_data["forecasted_errors"] = pd.Series(forecast)
predict_data["improved"] = predict_data["forecasted_sales"]+predict_data["forecasted_errors"]
predict_data["forecasted_sales"].plot()
predict_data["improved"].plot()
