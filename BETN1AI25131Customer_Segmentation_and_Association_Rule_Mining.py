#!/usr/bin/env python
# coding: utf-8

# # Clustering and Association Rule Mining
# 
# ## Objective
# To understand and implement clustering and association
# rule mining algorithms using real-world datasets.
# 
# ## Algorithms Used
# 1. K-Means Clustering
# 2. Hierarchical Clustering
# 3. DBSCAN
# 4. Apriori
# 5. FP-Growth

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv("Mall_Customers.csv")


# In[7]:


df.head(10)


# In[8]:


df.tail(10)


# In[9]:


print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# In[10]:


df.describe()


# In[11]:


df.describe(include='all')


# In[12]:


print(df.isnull().sum())


# In[13]:


print("Total Missing Values:", df.isnull().sum().sum())


# In[14]:


features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

df[features].hist(
    figsize=(12, 8),
    bins=20,
    edgecolor='black'
)

plt.suptitle("Distribution of Customer Features")
plt.tight_layout()
plt.show()


# In[15]:


plt.figure(figsize=(12, 6))

sns.boxplot(data=df[features])

plt.title("Boxplot of Customer Features")
plt.xticks(rotation=15)
plt.show()


# In[16]:


plt.figure(figsize=(6, 4))

sns.countplot(data=df, x='Gender')

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


# In[17]:


df.info()


# In[18]:


data = df.copy()

print("Dataset copied successfully!")
print("Dataset Shape:", data.shape)


# In[19]:


data = data.drop(columns=['CustomerID'])

print("CustomerID column removed successfully!")
print(data.head())


# In[20]:


print(data.isnull().sum())


# In[21]:


print("Total Missing Values:", data.isnull().sum().sum())


# In[23]:


for col in data.columns:

    if data[col].isnull().sum() > 0:

        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])

        else:
            data[col] = data[col].fillna(data[col].median())



# In[24]:


print(data.isnull().sum())


# In[25]:


features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

X = data[features]

print(X.head())


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[28]:


X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

print(X_scaled.head())


# In[29]:


print("Mean of Scaled Features:")
print(X_scaled.mean())

print("\nStandard Deviation:")
print(X_scaled.std())


# In[30]:


from sklearn.cluster import KMeans


# In[31]:


wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

print("WCSS values:", wcss)


# In[32]:


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o',
    linestyle='--'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid()
plt.show()


# In[37]:


kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# In[38]:


data['Cluster'] = clusters

print(data.head(10))


# In[39]:


plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Cluster'],
    palette='viridis',
    s=80
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()


# In[40]:


print(data['Cluster'].value_counts().sort_index())


# In[41]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


# In[42]:


plt.figure(figsize=(12, 6))

linked = linkage(X_scaled, method='ward')

dendrogram(linked)

plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")

plt.show()


# In[44]:


hc_model = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'
)

hc_clusters = hc_model.fit_predict(X_scaled)


# In[45]:


data['Hierarchical_Cluster'] = hc_clusters

print(data.head(10))


# In[46]:


plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['Hierarchical_Cluster'],
    palette='viridis',
    s=80
)

plt.title("Hierarchical Clustering - Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()


# In[47]:


print("K-Means Cluster Count:")
print(data['Cluster'].value_counts().sort_index())

print("\nHierarchical Cluster Count:")
print(data['Hierarchical_Cluster'].value_counts().sort_index())


# In[48]:


from sklearn.cluster import DBSCAN


# In[49]:


dbscan_model = DBSCAN(
    eps=0.5,
    min_samples=5
)

dbscan_clusters = dbscan_model.fit_predict(X_scaled)


# In[50]:


data['DBSCAN_Cluster'] = dbscan_clusters

print(data[['Age', 'Annual Income (k$)',
            'Spending Score (1-100)',
            'DBSCAN_Cluster']].head(10))


# In[51]:


plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=data['Annual Income (k$)'],
    y=data['Spending Score (1-100)'],
    hue=data['DBSCAN_Cluster'],
    palette='viridis',
    s=80
)

plt.title("Customer Segmentation using DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.show()


# In[52]:


print("DBSCAN Cluster Analysis")

print(data['DBSCAN_Cluster'].value_counts().sort_index())

n_clusters = len(set(dbscan_clusters)) - (
    1 if -1 in dbscan_clusters else 0
)

n_noise = list(dbscan_clusters).count(-1)

print("\nNumber of Clusters:", n_clusters)
print("Number of Noise Points:", n_noise)


# In[53]:


print("K-Means Clusters:",
      len(set(clusters)))

print("Hierarchical Clusters:",
      len(set(hc_clusters)))

print("DBSCAN Clusters:",
      n_clusters)

print("DBSCAN Noise Points:",
      n_noise)


# In[55]:


get_ipython().system('pip install mlxtend')


# In[56]:


from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# In[58]:


groceries = pd.read_csv("Groceries_dataset.csv")


# In[59]:


groceries.head(10)


# In[60]:


groceries.tail(10)


# In[62]:


groceries.info()


# In[63]:


print(groceries.isnull().sum())

print("Total Missing Values:",
      groceries.isnull().sum().sum())


# In[64]:


print(groceries.columns)


# In[65]:


print(groceries.columns)
print(groceries.head())


# In[66]:


transactions = groceries.groupby(
    ['Member_number', 'Date']
)['itemDescription'].apply(list)

print(transactions.head())


# In[67]:


transaction_list = transactions.tolist()

print("Total Transactions:", len(transaction_list))

print("\nFirst 5 Transactions:")
for i in transaction_list[:5]:
    print(i)


# In[72]:


from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_array = te.fit(transaction_list).transform(transaction_list)

basket = pd.DataFrame(
    te_array,
    columns=te.columns_
)
basket.head()


# In[70]:


print("Basket Shape:", basket.shape)

print("\nTotal Products:", len(basket.columns))

print("\nTotal Transactions:", len(basket))


# In[73]:


from mlxtend.frequent_patterns import apriori, association_rules
import time


# In[74]:


start_time = time.time()

frequent_itemsets_apriori = apriori(
    basket,
    min_support=0.01,
    use_colnames=True
)

apriori_time = time.time() - start_time

print("Apriori Algorithm Completed Successfully!")
print("Execution Time:", apriori_time, "seconds")


# In[75]:


frequent_itemsets_apriori.head(10)


# In[80]:


rules_apriori = association_rules(
    frequent_itemsets_apriori,
    metric="confidence",
    min_threshold=0.1
)


# In[78]:


rules_apriori[
    ['antecedents', 'consequents',
     'support', 'confidence', 'lift']
].head(10)


# In[81]:


print("Total Frequent Itemsets:",
      len(frequent_itemsets_apriori))

print("Total Association Rules:",
      len(rules_apriori))

print("Apriori Execution Time:",
      apriori_time, "seconds")


# In[84]:


start_time = time.time()

frequent_itemsets_fp = fpgrowth(
    basket,
    min_support=0.01,
    use_colnames=True
)

fp_time = time.time() - start_time

print("Execution Time:", fp_time, "seconds")


# In[83]:


frequent_itemsets_fp.head(10)


# In[85]:


rules_fp = association_rules(
    frequent_itemsets_fp,
    metric="confidence",
    min_threshold=0.1
)


# In[88]:


rules_fp[
    ['antecedents', 'consequents',
     'support', 'confidence', 'lift']
].head(10)


# In[89]:


print("Total Frequent Itemsets:",
      len(frequent_itemsets_fp))

print("Total Association Rules:",
      len(rules_fp))

print("FP-Growth Execution Time:",
      fp_time, "seconds")


# In[90]:


comparison = pd.DataFrame({
    'Algorithm': ['Apriori', 'FP-Growth'],

    'Frequent Itemsets': [
        len(frequent_itemsets_apriori),
        len(frequent_itemsets_fp)
    ],

    'Association Rules': [
        len(rules_apriori),
        len(rules_fp)
    ],

    'Average Support': [
        rules_apriori['support'].mean(),
        rules_fp['support'].mean()
    ],

    'Average Confidence': [
        rules_apriori['confidence'].mean(),
        rules_fp['confidence'].mean()
    ],

    'Average Lift': [
        rules_apriori['lift'].mean(),
        rules_fp['lift'].mean()
    ],

    'Execution Time': [
        apriori_time,
        fp_time
    ]
})

comparison


# In[91]:


plt.figure(figsize=(7, 5))

sns.barplot(
    data=comparison,
    x='Algorithm',
    y='Execution Time'
)

plt.title("Apriori vs FP-Growth Execution Time")
plt.ylabel("Time (Seconds)")
plt.show()


# In[92]:


comparison.set_index('Algorithm')[
    ['Average Support', 'Average Confidence', 'Average Lift']
].plot(kind='bar', figsize=(9, 5))

plt.title("Apriori vs FP-Growth Metrics Comparison")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.legend()
plt.show()


# In[94]:


apriori_report = rules_apriori.copy()
apriori_report['Algorithm'] = 'Apriori'

fp_report = rules_fp.copy()
fp_report['Algorithm'] = 'FP-Growth'

association_report = pd.concat(
    [apriori_report, fp_report],
    ignore_index=True
)

association_report.to_csv(
    "Association_Rule_Report.csv",
    index=False
)


# ## Conclusion
# 
# In this practical I learned about Clustering and Association Rule Mining using two datasets
# 
# First I used the Mall Customer Dataset and cleaned the data. Then I used K Means Hierarchical Clustering and DBSCAN algorithms. I used the Elbow Method and selected 4 clusters for K Means. Hierarchical Clustering also gave 4 clusters. DBSCAN found 6 clusters and 60 noise points.
# 
# After that I used the Groceries Dataset. I applied Apriori and FP Growth algorithms to find products which people buy together. Both algorithms gave 69 frequent itemsets and 4 association rules.
# 
# I also compared Support Confidence Lift and Execution Time. Apriori took 0.248 seconds and FP Growth took 0.392 seconds.
# 
# Through this practical I learned how to group customers and find shopping patterns using Machine Learning algorithms.

# In[ ]:




