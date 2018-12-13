import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("MY2017FuelConsumptionRatings.csv")

print(df.head())
print(df.describe())

cdf = df[["ENGINE SIZE","FUEL CONSUMPTION","CO2 EMISSIONS "]]
print(cdf.head())

plt.scatter(cdf["ENGINE SIZE"],cdf["CO2 EMISSIONS "],color="Red")
plt.show()

msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]
test = cdf[~msk]

from sklearn import linear_model

reg = linear_model.LinearRegression()

train_x = np.asanyarray(train[["ENGINE SIZE"]])
train_y = np.asanyarray(train[["CO2 EMISSIONS "]])

reg.fit(train_x,train_y)

print("Coef : " + str(reg.coef_))
print("Intercept : " + str(reg.intercept_))
