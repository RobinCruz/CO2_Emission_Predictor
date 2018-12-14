#the # lines of code where for my insight in the data to choose what to use for prediction :)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("MY2017FuelConsumptionRatings.csv")

#print(df.head())
#print(df.describe())

cdf = df[["ENGINE SIZE","FUEL CONSUMPTION","CO2 EMISSIONS"]]
print(cdf.head())

#vis = cdf[["ENGINE SIZE","FUEL CONSUMPTION","CO2 EMISSIONS"]]
#plt.hist(vis,10)
#plt.show()

#plt.scatter(cdf["ENGINE SIZE"],cdf["CO2 EMISSIONS"],color = "green")
#plt.xlabel("Engine Size")
#plt.ylabel("CO2")
#plt.show()

#plt.scatter(cdf["FUEL CONSUMPTION"],cdf["CO2 EMISSIONS"],color = "green")
#plt.xlabel("Fuel Consumption")
#plt.ylabel("CO2")
#plt.show()

#Splitting Data
msk =np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#plt.scatter(train["ENGINE SIZE"],train["CO2 EMISSIONS"],color = "green")
#plt.xlabel("Engine Size")
#plt.ylabel("CO2")
#plt.show()




#Since I felt Fuel Consumption was More Realating To CO2 Emmissions
from sklearn import linear_model

reg = linear_model.LinearRegression()

train_x = np.asanyarray(train[['FUEL CONSUMPTION']])
train_y = np.asanyarray(train[['CO2 EMISSIONS']])
reg.fit(train_x,train_y)

print("slope:")
print(reg.coef_)
print("const:")
print(reg.intercept_)

plt.scatter(train_x,train_y,color = "red")
plt.plot(train_x,reg.coef_[0][0]*train_x+reg.intercept_[0],"-b")
plt.xlabel("Fuel Consumption")
plt.ylabel("CO2")
plt.show()

#Predicting

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[["FUEL CONSUMPTION"]])
test_y = np.asanyarray(test[["CO2 EMISSIONS"]])

Pred = reg.predict(test_x)

print("Mean Absolute Error: %2f"%np.mean(np.absolute(Pred - test_y)))
print("Mean Squared Error: %2f"%np.mean((Pred - test_y)**2))
print("R2_Score: %2f"%r2_score(Pred,test_y))
