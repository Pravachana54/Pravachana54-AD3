#!/usr/bin/env python
# coding: utf-8

# # Electric power consumption (kWh per capita)

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# dataset : https://data.worldbank.org/indicator/EG.USE.ELEC.KH.PC


# In[2]:


def worldbank_data(path): 
    """
    This function reads the Electric power consumption (kWh per capita) data in csv file

    Returns:
    - Original Dataset
    - Transposed Dataset

    """
    
    countries = pd.read_csv(path, skiprows=4).iloc[:, :-1]
    years = pd.read_csv(path, skiprows=4)
    years = years.set_index(['Country Name']).T.iloc[:, :-1]
    return countries, years


# In[3]:


countries_df, years_df = worldbank_data('API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_6300057.csv')


# ## Clustering

# In[4]:


countries_df.head()


# In[5]:


data = countries_df.filter(['Country Name'] + [str(year) for year in range(2000, 2015)])


# In[6]:


data.head()


# In[7]:


data = data.dropna()


# In[8]:


subset = data[["Country Name", "2014"]].copy()


# In[9]:


subset['Change (%)'] = 100.0 * ((data['2014'] - data["2001"]) / data['2001'])


# In[10]:


subset.describe()


# In[11]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset, x="2014", y="Change (%)", label="Electric power consumption (kWh per capita)", color="red")
plt.title("Scatter Plot")
plt.xlabel("Electric power consumption (kWh per capita) in 2014")
plt.ylabel("Change (%) from 2001 to 2014")
plt.show()


# In[12]:


import sklearn.preprocessing as pp
x = subset[["2014", "Change (%)"]].copy()
scaler = pp.RobustScaler()
scaler.fit(x)
x_norm = scaler.transform(x)


# In[13]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=x_norm, x=x_norm[:, 0], y=x_norm[:, 1], label="Electric power consumption (kWh per capita)", color="red")
plt.xlabel("Normalized Electric power consumption (kWh per capita)")
plt.ylabel("Normalized Change (%)")
plt.title("Normalized Data")
plt.show()


# In[14]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[38]:


from sklearn.cluster import KMeans
import sklearn.metrics as skmet

for i in range(2, 12):
    score = one_silhouette(x_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[39]:


kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
kmeans.fit(x_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[43]:


plt.figure(figsize=(10, 6))
plt.scatter(subset["2014"], subset["Change (%)"], c=labels, s=10)
plt.scatter(xkmeans, ykmeans, marker="x", c="red", s=50, label="Cluster Centroids")
plt.title("Clusters")
plt.xlabel("Electric power consumption (kWh per capita) in 2014")
plt.ylabel("Change (%) from 2001 to 2020")
plt.legend()
plt.show()


# ## Curve Fitting

# In[15]:


def preprocess_uk_power_consumption(years_df, start_year, end_year, column_name):
    """
    Preprocess the UK electric power consumption data from the given DataFrame.
    Returns:
    - DataFrame: Processed DataFrame with specified subset, renamed column, and numeric data.
    """
    uk_pow = years_df.loc[start_year:end_year, [column_name]].reset_index().rename(columns={'index': 'Year', column_name: 'Electric power consumption (kWh per capita)'})
    uk_pow = uk_pow.apply(pd.to_numeric, errors='coerce')

    return uk_pow 

uk_pow = preprocess_uk_power_consumption(years_df, '2000', '2014', 'United Kingdom')
uk_pow.describe()


# In[16]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=uk_pow, x='Year', y='Electric power consumption (kWh per capita)')
plt.xlabel('Years')
plt.ylabel('Electric power consumption (kWh per capita)')
plt.title('Electric power consumption (kWh per capita) in United Kingdom between 2000-2014')
plt.show()


# In[47]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[53]:


import scipy.optimize as opt
import errors

param, covar = opt.curve_fit(poly, uk_pow["Year"], uk_pow["Electric power consumption (kWh per capita)"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2000, 2025)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
uk_pow["fit"] = poly(uk_pow["Year"], *param)


# In[57]:


plt.figure(figsize=(10, 6))
plt.plot(uk_pow["Year"], uk_pow["Electric power consumption (kWh per capita)"], label="Electric power consumption (kWh per capita)")
plt.plot(year, forecast, label="forecast", c= 'red')
plt.fill_between(year, low, up, color="green", alpha=0.7)
plt.title("Electric power consumption (kWh per capita) Prediction in United Kingdom")
plt.xlabel("Year")
plt.ylabel("Electric power consumption (kWh per capita)")
plt.legend()
plt.show()


# In[ ]:




