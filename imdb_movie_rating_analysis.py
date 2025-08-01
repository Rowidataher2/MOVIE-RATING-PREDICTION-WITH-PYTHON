#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv(r'C:\Users\ROWIDA\Documents\dataset_2\IMDb_Movies_India.csv', encoding='latin1')
data


# In[57]:


data.shape


# In[58]:


data.describe(include='all')


# In[59]:


data.dtypes


# In[60]:


data.info()


# In[61]:


data.isnull().values.any()


# In[62]:


data.isnull().sum()


# In[63]:


data.isnull().values.any()


# In[65]:


data.nunique()


# In[66]:


data.dropna(axis=0,inplace=True)
data


# In[67]:


data.duplicated().any()


# In[68]:


data.drop_duplicates()


# In[70]:


data.describe(include = 'all')


# In[71]:


data['Duration'] = data['Duration'].str.extract('(\d+)')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')


# In[72]:


data["Votes"]=data["Votes"].replace("$5.16M", 516)
data["Votes"] = pd.to_numeric(data['Votes'].str.replace(',',''))


# In[73]:


data['Year'] = data['Year'].str.extract('(\d+)') 
data['Year'] = pd.to_numeric(data['Year'], errors='coerce') 


# In[74]:


data["Year"].head()


# In[75]:


data.describe()


# In[76]:


data.dtypes


# In[77]:


from collections import Counter

genre=data['Genre']
genres=data['Genre'].str.split(',',expand=True)

genre_counts = Counter(genre for genre in genres.values.flatten() if genre is not None)

# Sort the genre counts 
sorted_genre_counts = dict(sorted(genre_counts.items()))

for genre, count in sorted_genre_counts.items():
    print(f"{genre}: {count}")


# In[78]:


generscount=data["Genre"].value_counts()
generscount.head(5)


# In[79]:


#total number of action in each column
total_action_count = (data['Genre'].str.contains('Action')).sum()
print(total_action_count)


# In[80]:


print(genres)


# In[81]:


#time of each director's name appear
directors = data["Director"].value_counts()
directors.head(5)


# In[82]:


#concatenating three columns ('Actor 1', 'Actor 2', 'Actor 3')
actors = pd.concat([data['Actor 1'], data['Actor 2'], data['Actor 3']]).dropna().value_counts()
actors.head(5)



# In[83]:


data.isnull().sum()


# In[84]:


data.isnull()


# In[85]:


data.dropna(subset=['Year'], inplace=True)


# In[86]:


data['Year'] = data['Year'].astype(int)


# In[87]:


data.dtypes


# In[89]:


sns.boxplot(data=data,x='Year')
#sns.lineplot(data=data['Year'].value_counts().sort_index())
plt.title("box plot for year")
plt.show()


# In[90]:


plt.plot(data["Year"].value_counts().sort_index(),data=data)
plt.xlabel("years")
plt.ylabel("count")
plt.title("Annual Movie Release Counts Over Time")
plt.show()


# In[91]:


import seaborn as sns
sns.lineplot(data=data.groupby('Year')['Duration'].mean().reset_index(), x='Year', y='Duration')
plt.xlabel("years")
plt.ylabel("Average Duration")
plt.title("Average Movie Duration Trends Over the Years")
plt.show()


# In[92]:


sns.boxplot(data=data, x='Duration')
plt.title("Box Plot of Average Movie Durations")
plt.show()


# In[95]:


Q1 = data['Duration'].quantile(0.25)
Q3 = data['Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Duration'] >= lower_bound) & (data['Duration'] <= upper_bound)]
#print(lower_bound)
##72.5 ,196.5
data.head()



# In[96]:


print(data["Duration"].max())
print(data["Duration"].min())
#max duration < upperbound
#min duration > lowerbound


# In[97]:


sns.boxplot(data=data, x='Rating')
plt.title("Box Plot of Movie Ratings")
plt.show()


# In[98]:


Q1 = data['Rating'].quantile(0.25)
Q3 = data['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Rating'] >= lower_bound) & (data['Rating'] <= upper_bound)]
data.head()


# In[99]:


print(data["Rating"].max())
print(data["Rating"].min())


# In[100]:


rating_of_votes = data.groupby('Rating')['Votes'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=rating_of_votes,x='Rating',y='Votes')
plt.xlabel('Rating')
plt.ylabel('Total Votes')
plt.title('Total Votes per Rating')
plt.show()


# In[ ]:





# In[101]:


plt.figure(figsize=(15, 8))
sns.barplot(x=directors.head(30).index, y=directors.head(30).values, palette='viridis')
plt.xlabel('Directors')
plt.ylabel('Frequency of Movies')
plt.title('Top 30 Directors by Frequency of Movies')
plt.xticks(rotation=90)

plt.show()


# In[102]:


plt.figure(figsize=(15, 8))
sns.barplot(x=actors.head(30).index, y=actors.head(30).values, palette='viridis')
plt.xlabel('Actors')
plt.ylabel('Total Number of Movies')
plt.title('Top 30 Actors with Total Number of Movies')
plt.xticks(rotation=90)
plt.show()



# In[103]:


data["Directors"] = data['Director'].astype('category').cat.codes
data["Actor"] = data['Actor 1'] + ', ' + data['Actor 2'] + ', ' + data['Actor 3']
data["Actors"] = data['Actor'].astype('category').cat.codes
data["Genres"] = data['Genre'].astype('category').cat.codes
data.head()


# In[104]:


Q1 = data['Genres'].quantile(0.25)
Q3 = data['Genres'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Genres'] >= lower_bound) & (data['Genres'] <= upper_bound)]
data.head()


# In[105]:


sns.boxplot(data=data, x='Actors')
plt.title('Box Plot of Actors')
plt.show()


# In[106]:


Q1 = data['Actors'].quantile(0.25)
Q3 = data['Actors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Actors'] >= lower_bound) & (data['Actors'] <= upper_bound)]
data.head()



# In[107]:


sns.boxplot(data=data, x='Directors')
plt.title('Box Plot of Directors')
plt.show()


# In[108]:


Q1 = data['Directors'].quantile(0.25)
Q3 = data['Directors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Directors'] >= lower_bound) & (data['Directors'] <= upper_bound)]
data.head(5)


# In[109]:


sns.histplot(data = data, x = "Rating", bins = 20, kde = True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Movie Ratings')
plt.show()


# In[110]:


#SPLITTING THE DATA
from sklearn.model_selection import train_test_split
Input = data.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor'], axis=1)
Output = data['Rating']



# In[111]:


Input.head()


# In[112]:


x_train, x_test, y_train, y_test = train_test_split(Input, Output, test_size = 0.2, random_state = 1)


# In[113]:


from sklearn.metrics import mean_squared_error, r2_score as score
def evaluate_model(y_true, y_pred, model_name):
    print("Model: ", model_name)
    print("Accuracy = {:0.2f}%".format(score(y_true, y_pred)*100))
    print("Mean Squared Error = {:0.2f}\n".format(mean_squared_error(y_true, y_pred, squared=False)))
    return round(score(y_true, y_pred)*100, 2)



# In[114]:


#Create a LinearRegression 
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

# Fit the model
LR.fit(x_train, y_train)

# Make predictions 
lr_preds = LR.predict(x_test)
LRScore = evaluate_model(y_test, lr_preds, "LINEAR REGRESSION")
#print('accuracy: {:.2f}%'.format(LRScore*100))



# In[115]:


from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(random_state=1)
DTR.fit(x_train, y_train)
dt_preds = DTR.predict(x_test)
DTScore = evaluate_model(y_test, dt_preds, "DECEISION TREE")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




