#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
from datetime import datetime
sb.set() # set the default Seaborn style for graphics


# ## Data Exploration and Analysis:

# In[2]:


df = pd.read_csv('data/archive/retail_store_inventory.csv',  parse_dates=['Date'])
df.rename(columns={'Store ID': 'Store', 'Product ID': 'Item', 'Units Sold': 'Sales'}, inplace=True)
df.head()


# In[3]:


df.count()


# In[4]:


df["Store"].unique()


# In[5]:


df.describe()


# In[6]:


store1_df = pd.DataFrame(df[df["Store"] == "S001"][["Inventory Level","Sales", "Demand Forecast", "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", "Units Ordered"]])
store2_df = pd.DataFrame(df[df["Store"] == "S002"][["Inventory Level","Sales", "Demand Forecast", "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", "Units Ordered"]])
store3_df = pd.DataFrame(df[df["Store"] == "S003"][["Inventory Level","Sales", "Demand Forecast", "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", "Units Ordered"]])
store4_df = pd.DataFrame(df[df["Store"] == "S004"][["Inventory Level","Sales", "Demand Forecast", "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", "Units Ordered"]])
store5_df = pd.DataFrame(df[df["Store"] == "S005"][["Inventory Level","Sales", "Demand Forecast", "Price", "Discount", "Holiday/Promotion", "Competitor Pricing", "Units Ordered"]])

f, axes = plt.subplots(2, 2, figsize = (15,15))
sb.heatmap(store1_df.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f", ax=axes[0, 0])
sb.heatmap(store2_df.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axes[0, 1])
sb.heatmap(store3_df.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axes[1, 0])
sb.heatmap(store4_df.corr(), vmin=-1, vmax=1, annot=True, fmt=".2f", ax=axes[1, 1])


# In[7]:


sb.heatmap(store5_df.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f")


# In[8]:


df['Year'] = df['Date'].dt.year
year2022 = df[df["Year"] == 2022]

plt.figure(figsize=(14,6))
sb.lineplot(data=year2022, x='Date', y='Sales', hue='Store', ci=None)
plt.title('Sales Over Time by Store for 2022')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(title='Store')
plt.tight_layout()
plt.show()


# In[9]:


year2023 = df[df["Year"] == 2023]

plt.figure(figsize=(14,6))
sb.lineplot(data=year2023, x='Date', y='Sales', hue='Store', ci=None)
plt.title('Sales Over Time by Store for 2023')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(title='Store')
plt.tight_layout()
plt.show()


# In[10]:


plt.figure(figsize=(8,5))
sb.histplot(df['Sales'], bins=30, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Units Sold')
plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(14,6))
sb.lineplot(data=df, x='Date', y='Sales', hue='Item', style='Store', estimator=None)
plt.title('Sales Over Time by Store and Item')
plt.xlabel('Date')
plt.ylabel('Units Sold')
plt.tight_layout()
plt.show()


# In[12]:


def assign_period(date):
    if date.month <= 6:
        if date.year == 2022:
            return 'Period 1 (Jan-Jun 2022)'
        elif date.year == 2023:
            return 'Period 3 (Jan-Jun 2023)'
        else:
            return 'Period 4 (Jul-Dec 2023 + 01 Jan 2024)'
    else:
        if date.year == 2022:
            return 'Period 2 (Jul-Dec 2022)'
        elif date.year == 2023:
            return 'Period 4 (Jul-Dec 2023 + 01 Jan 2024)'


# In[13]:


df['Period'] = df['Date'].apply(assign_period)


# In[14]:


stores = df['Store'].unique()
items = df['Item'].unique()
periods = df['Period'].unique()
for store in stores:
    for item in items:
        for period in periods:
            store_item_data = df[(df["Store"] == store) & (df["Item"] == item) & (df['Period'] == period)]
            if not store_item_data.empty:
                plt.figure(figsize=(14, 6))
                sb.lineplot(data=store_item_data, x='Date', y='Sales', estimator=None)
                
                plt.title(f'Sales Over {period} by {store} and {item}')
                plt.xlabel('Date')
                plt.ylabel('Units Sold')
                plt.tight_layout()
                plt.show()
            else:
                print(f"No data for {store}, {item}, {period}")


# In[15]:


product_performance = df.groupby('Item')['Sales'].agg(['mean','std','sum'])
product_performance.columns = ['Average Sales','Sales Variability','Total Sales']
print("\nTop Performing Products:")
display(product_performance.sort_values('Total Sales', ascending=False).head(20))


# ## Predictive Modelling:

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[17]:


def linearReg(xTrain, xTest, yTrain, yTest):
    lr = LinearRegression()
    lr.fit(xTrain, yTrain)
              
    yPred = lr.predict(xTest)
    score = lr.score(xTest,yTest)
    
    mse = mean_squared_error(yTest, yPred, squared=False)
    rmse = mean_squared_error(yTest, yPred, squared=True)
    mae = mean_absolute_error(yTest, yPred)
    return score, mse, rmse, mae


# In[18]:


def randomForestReg(xTrain, xTest, yTrain, yTest):
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(xTrain, yTrain)
              
    yPred = rfr.predict(xTest)
    score = rfr.score(xTest,yTest)
    
    mse = mean_squared_error(yTest, yPred, squared=False)
    rmse = mean_squared_error(yTest, yPred, squared=True)
    mae = mean_absolute_error(yTest, yPred)
    return score, mse, rmse, mae


# In[19]:


x = df[['Price', 'Discount', 'Inventory Level', 'Units Ordered', 'Demand Forecast']]
y = df['Sales']


# In[20]:


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

i = 0
lrMse = 0
lrMae = 0
lrRmse = 0
lrScore = 0

rfrMse = 0
rfrMae = 0
rfrRmse = 0
rfrScore = 0

while (i != 50):
    lrResult = linearReg(xTrain, xTest, yTrain, yTest)
    lrScore += lrResult[0]
    lrMse += lrResult[1]
    lrRmse += lrResult[2]
    lrMae += lrResult[3]
    rfrResult = randomForestReg(xTrain, xTest, yTrain, yTest)
    rfrScore += rfrResult[0]
    rfrMse += rfrResult[1]
    rfrRmse += rfrResult[2]
    rfrMae += rfrResult[3]
    print(i)
    i += 1


# In[21]:


print("Average (over 50 times) Linear Regression MAE:", lrMae/i)
print("Average (over 50 times) Linear Regression RMSE:", lrRmse/i)
print("Average (over 50 times) Linear Regression MSE:", lrMse/i)
print("Average (over 50 times) Linear Regression Accuracy:", lrScore/i)

print("Average (over 50 times) Random Forest Regression MAE:", rfrMae/i)
print("Average (over 50 times) Random Forest Regression RMSE:", rfrRmse/i)
print("Average (over 50 times) Random Forest Regression MSE:", rfrMse/i)
print("Average (over 50 times) Random Forest Regression Accuracy:", rfrScore/i)


# In[ ]:




