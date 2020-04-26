#!/usr/bin/env python
# coding: utf-8

# In[213]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[214]:


house_file_path="E:\\Machine_learning_projects\\proj1\\housing.csv"


# In[160]:


housing=pd.read_csv(house_file_path)


# In[215]:


housing[:5]


# In[219]:


housing.describe()


# In[162]:


housing.info()


# In[163]:


housing["ocean_proximity"].value_counts()


# In[164]:


housing["longitude"].max()


# In[165]:


housing["longitude"].min()


# In[166]:


housing.hist()


# In[172]:


import seaborn as sns
sns.scatterplot(x=housing["longitude"],y=housing["latitude"])


# In[173]:


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)


# In[174]:


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)


# In[175]:


#corroleation
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[176]:


housing.plot(kind="scatter",x="median_house_value",y="median_income",alpha=0.1)


# In[177]:


import seaborn as sns
sns.regplot(x=housing["median_house_value"],y=housing["median_income"])


# In[167]:


#first approach
#determined train ,test set by your self 
trainTOtest=0.8
import numpy as np 
Ind=np.random.permutation(len(housing))
print(Ind)
triantotestsize=int(trainTOtest*len(housing))
print(triantotestsize)
trainSet=housing[:triantotestsize]
testSet=housing[triantotestsize:]


# In[168]:


#second approach
##data cleaning
#median=housing["total bedrooms"].median()
#housing["total_bedrooms"].fillna(median,inpplace=True)
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy="median")
housingFilterd=housing.drop("ocean_proximity",axis=1)
housing_cat=housing[["ocean_proximity"]]
imputer.fit(housingFilterd)
X=imputer.transform(housingFilterd)
housingX=pd.DataFrame(X,columns=housingFilterd.columns)


# In[ ]:


#handle text catigorical
#conveert text to numirical values using LabelEncoder

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
housing_cat_1hot=encoder.fit_transform(housing_cat)
#housing_cat_1hot
housingX["ocean_proximity"]=housing_cat_1hot.ravel()
#housingX


# In[170]:


#feature Scaling 
#from sklearn.preprocessing import StandardScaler
#std_scale=StandardScaler()
#XX=std_scale.fit_transform(housingX)
#housingX=pd.DataFrame(XX,columns=housingX.columns)
#housingX


# In[220]:


#second approach
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housingX,test_size=0.2,random_state=42)


# In[221]:


train_set[:5]


# In[222]:


test_set[:5]


# In[225]:


print(train_set.shape)
print(test_set.shape)


# In[178]:


house=housing.copy()
house["rooms_per_household"]=housing["total_rooms"]/housing["households"]
house["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
house["population_per_household"]=housing["population"]/housing["households"]


# In[179]:


corr_matrix=house.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[180]:


trainx=train_set.drop("median_house_value",axis=1)
trainy=train_set["median_house_value"]
testx=test_set.drop("median_house_value",axis=1)
testy=test_set["median_house_value"]


# In[181]:


trainx


# In[182]:


#train model using LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg=LinearRegression()
lin_reg.fit(trainx,trainy)

#evaluate error
y_pred=lin_reg.predict(trainx)
lin_rms=mean_squared_error(trainy,y_pred)
lin_rmse=np.sqrt(lin_rms)
lin_rmse


# In[183]:


#train model 
#using DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(trainx,trainy)


# In[184]:


#evaluate error
house_predect=tree_reg.predict(trainx)
tree_rms=mean_squared_error(trainy,house_predect)
tree_rmse=np.sqrt(tree_rms)
tree_rmse


# In[185]:


#better evaluation using cross validation 
from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,trainx,trainy,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
rmse_scores


# In[188]:


print(rmse_scores.mean())
print(rmse_scores.std())


# In[ ]:


#using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(trainx,trainy)
ho_predect=forest_reg.predict(trainx)
forest_rms=mean_squared_error(trainy,ho_predect)
forest_rmse=np.sqrt(forest_rms)


# In[197]:


forest_rmse


# ## Fine-Tune the model

# In[203]:


#using grid search
from sklearn.model_selection import GridSearchCV
param_grid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]
forest_reg=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error')
grid_search.fit(trainx,trainy)


# In[205]:


grid_search.best_params_


# In[207]:


grid_search.best_estimator_


# In[212]:


final_model=grid_search.best_estimator_
final_pred=final_model.predict(testx)
final_mse=mean_squared_error(testy,final_pred)
final_rmse=np.sqrt(final_mse)
final_rmse


# In[ ]:




