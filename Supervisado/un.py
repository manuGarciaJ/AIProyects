#!/usr/bin/env python
# coding: utf-8

# # Unsupervised learning algorithms
# 
# Here I have implemeted a compilation of clustering algorithms, most of them taken from: 

# In[1]:


import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import umap
from sklearn.manifold import TSNE


# ## Creating dummy data

# In[140]:


clus1 = np.random.multivariate_normal(mean=[0,0],cov=[[1,0],[0,1]], size= 50)
clus2 = np.random.multivariate_normal(mean=[5,5],cov=[[1,0],[0,1]], size= 50)
clus3 = np.random.multivariate_normal(mean=[10,10],cov=[[1,0],[0,1]], size= 50)
dummy= pd.DataFrame(np.concatenate((clus1,clus2,clus3)), columns=['x','y'])
#dummy.plot.scatter(x='x',y='y',title='Dummy data')


# In[4]:


#dummy.head()


# ## Splitting data into 3

# In[5]:


def split(data):
    train= data.sample(frac=0.8)
    val= data.drop(train.index)
    test= train.sample(frac=0.25)
    train= train.drop(test.index)
    return (train,test,val)

aux=split(dummy)


# ## Normalization

# In[6]:


def normalize(df):
    return (df-df.mean())/(df.max()-df.min())

#normalize(dummy)


# ## A couple of distances

# In[7]:


#euclidean
def eucli(p1,p2,data):
    d=0
    for i in p1.index: #so that it works on n dimentions.
        d+=(p1[i]-p2[i])**2
    return np.sqrt(d)


# In[8]:


#mahalonobis
def mahala(p1,p2,data):
    cova= data.cov()
    invcov= pd.DataFrame(np.linalg.pinv(cova.values), cova.columns, cova.index)
    subs= p1-p2
    return subs.dot(invcov).dot(subs.T)

#example of use: mahala(dummy.iloc[0],dummy.iloc[1],dummy)


# In[9]:


#manhattan
def manhat(p1,p2,data):
    d=0
    for i in p1.index: #so that it works on n dimentions.
        d+=abs(p1[i]-p2[i])
    return d


# ## Clustering techniques from (Hammouda,2000)

# ### Clustering
# 
# Here we follow the regular k-means clustering algorithm with the initial set of centroids taken as input. K-means, mountain, substractive and our clustering method all follow this presidure after selecting the initial centroids in their own way.

# In[10]:


def cluster(df,c,k,dist,maxiter):
    #0)initialize variables
    Z= 10**15
    n= df.shape[0]  
    #1) Initial centroids are chosen by other algorithms.
    for l in range(maxiter):
        #2)memebership matrix and cost function
        uaux= [-1 for i in range(n)]
        u= pd.DataFrame([uaux for i in range(k)]).T
        J=0
        for i in range(n):
            distances=[0]*k
            for j in range(k):
                #print('i= '+str(i))
                #print('j= '+str(j))
                d=dist(df.loc[i],c.loc[j],df)
                distances[j]= d
            m= min(distances)
            J+=m #put distance of cluster x_i belongs to into cost function (x_i belongs to the 
            #cluster who's centroid it has closest.)
            u.loc[i] = [1 if distances[i]==m else 0 for i in range(k)] # 1 for the cluster represented 
            #by the closest centroid and 0 for the others.

        # 3) See if stop criteria is met
        print(Z)
        if abs(J-Z)/J<0.01:#((abs(J-Z)/J<0.01) or (J>Z)):
            df['label']= u.idxmax(axis=1)
            print(J)
            print(l) #uncomment if you wanna see the number of iterations before reaching end criteria
            return (df,c)
        else: Z=J
            
        # 4) Recalculate the centroids before iterating again
        df['label']= u.idxmax(axis=1)
        c= df.groupby('label').mean()
        if c.shape[0]!=k: k=c.shape[0]
        #print(c)
        df= df.drop('label',axis=1)
    df['label']= u.idxmax(axis=1)
    print(J)
    return (df,c)


# ### Kmeans

# In[143]:


def k(df,k,dist,maxiter): #dist one of 'euclidean','mahalanobis' or 'manhattan'
    df= df.copy().reset_index(drop=True)   
    #1)initial centroids
    c= df.sample(n=k).reset_index(drop=True)   
    (df,c) = cluster(df,c,k,dist,maxiter)
    return df,c

# kmeaneados,c=k(dummy,3,manhat,100)
# sns.scatterplot(data=kmeaneados, x="x", y="y", hue="label")
# ax = plt.gca()
# ax.set_title("Dummy data clustered by working algorithm")


# ### Fuzzy C-means 
# 
# This one is interesting because it doesn't use the same process as the rest. Here we calculate an initial membership matrix and from it we find clusters. The other algorithms do it the other way around.

# In[12]:


def fcm(df,k,dist,maxiter): #dist is one of 'euclidean','mahalanobis' or 'manhattan'
    #0)initialize variables
    Z= 10**15
    n= df.shape[0]  
    df= df.copy().reset_index(drop=True)
    m=1.5
    #1)initial membership functions
    u=[[] for i in range(n)]
    for i in range(n):
        a = np.random.random(k)
        a /= a.sum()
        u[i]=a
    u= pd.DataFrame(u)
    #2)centroids and cost function
    for l in range(maxiter):
        c=pd.DataFrame([],columns=list(df.columns))
        for i in range(k):
            den= (u[i]**m).sum()
            num= pd.Series([0]*len(df.columns),index=df.columns)
            for j in range(150):
                num=num+(u.at[j,i]**m)*df.loc[j]
            c.loc[i]= num/den
        J=0
        distances= pd.DataFrame([],columns=list(range(k)))
        for i in range(n):
            daux=[0]*k
            for j in range(k):
                #print(c.loc[j])
                d=dist(df.loc[i],c.loc[j],df)
                daux[j]= d
                J+=(u.at[i,j]**m)*(d**2) #put distance into cost function
            distances.loc[i]=daux

            # 3) See if stop criteria is met
        #print(Z)
        if abs(J-Z)/Z<0.01:#((abs(J-Z)/Z<0.01) or (J<3)):
            df['label']= u.idxmax(axis=1)
            #print(J)
            #print(l) #uncomment if you wanna see the number of iterations before reaching end criteria
            return (df,c)
        else: Z=J
        # 4) Recalculate U before iterating again
        for i in range(k):
            for j in range(n):
                u.at[j,i]=0
                for o in range(k):
                    u.at[j,i]+=(distances.at[j,i]/distances.at[j,o])**(2/(m-1))
                u.at[j,i]=u.at[j,i]**(-1)
#         if u.isnull().sum().sum()>0:
#             df['label']= u.idxmax(axis=1)
#             return (df,c)
    df['label']= u.idxmax(axis=1)
    print(J)
    return (df,c)

# fuzzy,c=fcm(dummy,3,eucli,100) #
# sns.scatterplot(data=fuzzy, x="x", y="y", hue="label")


# ### Mountain

# In[74]:


def mount(df,k,dist,maxiter,sig=1.2,beta=1.2):
    den=3 #measure of grid density. For example, den=3 means that there are 3 possible 
    #values per veriable, creating a grid with 3**m points with m being the number of
    #variebles in the data
    #0) Initialize variables
    #sig=1.2
    #beta=1.2
    n,l=df.shape
    df= df.copy().reset_index(drop=True)
    # 1)Look for initial centroids
    if l==2:
        df.columns=['x','y']
        c=pd.DataFrame([],columns=['x','y'])
    else:
        df.columns=['x','y','z']
        c=pd.DataFrame([],columns=['x','y','z'])        
    tiles = [(1/den)*i-(0.5/den) for i in range(1,den+1)]
    #We will make DF containing the points separated by a column per variable
    gp= [[tiles[j]*(df[i].max()-df[i].min())+df[i].min() for j in range(den)] for i in df.columns]
    gp= pd.DataFrame(gp).T#,columns=range(den)
    gp.columns=df.columns
    #Here we list the points in the grid
    if l==2:
        grid= pd.DataFrame(list(product(gp['x'], gp['y'])),columns=['x','y'])
    else:
        grid= pd.DataFrame(list(product(gp['x'], gp['y'], gp['z'])),columns=['x','y','z'])
    gs= grid.shape[0]#amount of points in the grid
    m=[0]*gs
    for v in range(gs): # Here we find the initial values of mountain functions
        for i in range(n):
            m[v]+= np.exp(-(dist(grid.loc[v],df.loc[i],df)**2)/(2*sig**2))
    c.loc[0]= grid.loc[m.index(max(m))]
    #Here we find the rest of the centroids
    for i in range(1,k):
        for v in range(gs):
            m[v]= m[v]-max(m)*np.exp(-(dist(grid.loc[v],c.iloc[-1],grid)**2)/(2*beta**2))
        c.loc[i]= grid.loc[m.index(max(m))]
    #finally, we cluster. This should convert quicker than the other algorithms because 
    #of well chosen initial centroids.
    print(c)
    df,c= cluster(df,c,k,dist,maxiter)
    return df,c

#mount(normalize(umaptrain3D),5,manhat,100)


# ### Subtractive

# In[18]:


def sub(df,k,dist,maxiter,ra=1.5,rb=1.5):
    
    n=df.shape[0]
    df= df.copy().reset_index(drop=True)
    dm=pd.DataFrame(0.0, index=df.index, columns=df.index) #Distance matrix
    for i in range(n):
        for j in range(n):
            dm[i][j]= np.exp(-(dist(df.loc[i],df.loc[j],df)**2/(ra/2)**2))#
    dm['density']= dm.sum(axis=1)
    
    #select centroids:
    c= pd.DataFrame(0.0,index=range(k),columns=df.columns)
    for ki in range(k):
        nc= dm['density'].idxmax() #index new centroid
        c.loc[ki]= df.loc[nc]
        for i in range(n):
            dm.density[i] = dm.density[i]-dm.density[nc]*np.exp(-(dist(df.loc[i],c.loc[ki],df)**2/(rb/2)**2))
    
    df,c= cluster(df,c,k,dist,maxiter)
    return df,c

# s,c=sub(dummy,3,eucli,100)
# sns.scatterplot(data=s, x="x", y="y", hue="label")


# ## Initial centroids by percentiles
# 

# In[15]:


def per(df,k,dist,maxiter):
    df=df.copy().reset_index(drop=True)
    tiles = [(1/k)*i-(0.5/k) for i in range(1,k+1)] #percentiles I want for each centroid
    c=df.quantile(tiles)
    c.index=range(k)
    (df,c)= cluster(df,c,k,dist,maxiter)
    return (df,c)

# (perc,c)=per(dummy,3,eucli,100)
# sns.scatterplot(data=perc, x="x", y="y", hue="label")


# ## Now with real data

# Before we start, ance we have trained model, we want to be able to use it to label new data. In order to do this, we create the following predict function:

# In[19]:


def predict(df,c,dist):
    n= df.shape[0]
    k= c.shape[0]
    df= normalize(df.copy()).reset_index(drop=True)
    uaux= [-1 for i in range(n)]
    u= pd.DataFrame([uaux for i in range(k)]).T
    for i in range(n):
        distances=[0]*k
        for j in range(k):
            d=dist(df.loc[i],c.loc[j],df)
            distances[j]= d
        m= min(distances)
        u.loc[i] = [1 if distances[i]==m else 0 for i in range(k)] # 1 for the cluster represented 
    df['label']= u.idxmax(axis=1)
    return df

#predict(X_test,ct1,eucli)


# ### The data:

# In[20]:


# songs= pd.read_excel('dataSong.xlsx')
# songs.head()


# # We do some clean up. We get rid of some redundant columns and change the name column to the number of words in the title, as to see if the lenght of the name has any influence.

# # In[21]:


# songs.dropna(inplace=True)
# songs['index']=songs['index'].astype('int32')
# songs.set_index('index',inplace=True)
# songs.drop(['id','youtube','ytTitle','date','hiddenSubscriberCount','channelId','favoriteCount'],axis=1,inplace=True)
# songs['name']= songs['name'].str.split().apply(len)
# songs.head()


# # In[22]:


# songs.shape


# # ### Now we make our train, test, validation splits and normalize the train data.
# # 
# # The rest will only be normalized right before labeling once the models are tarined.

# # In[23]:


# (X_train,X_test,X_Val)= split(songs)


# # In[24]:


# X_train = normalize(X_train)
# X_train


# # ### UMAP
# # 
# # We use this to do dimension reduction. This is useful for plotting.

# # In[117]:


# reducer = umap.UMAP()
# umap_train = reducer.fit_transform(X_train)
# umap_train.shape


# # In[118]:


# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     )#c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})]
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of Spotify songs', fontsize=24)


# # From seeing this plot, I have arbitrarily decided to always use k=5 in all of our clustering ventures with this data.
# # ![alt text](arbi.jpg "Title")

# # In[27]:


# umaptrain=pd.DataFrame(umap_train, columns=['x','y'])


# # We repeate this for the test data so we can plot results taken from the predict function

# # In[112]:


# reducer = umap.UMAP()
# umap_test = reducer.fit_transform(normalize(X_test))
# umap_test.shape


# # In[113]:


# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     )#c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})]
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of Spotify songs test set', fontsize=24)


# # In[30]:


# umaptest=pd.DataFrame(umap_test, columns=['x','y'])


# # ### Let's see $R^n$ results

# # #### k-means

# # In[31]:


# keucli,ck1=k(X_train,5,eucli,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in keucli.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Euclidean distance', fontsize=12)


# # In[32]:


# #predict for the test data
# keuclitest= predict(X_test,ck1,eucli)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in keuclitest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Euclidean distance, test data', fontsize=12)


# # In[33]:


# kmahala,ck2=k(X_train,5,mahala,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in kmahala.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Mahalanobis distance', fontsize=12)


# # In[34]:


# #predict for the test data
# kmahalatest= predict(X_test,ck2,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in kmahalatest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Mahalanobis distance, test data', fontsize=12)


# # In[35]:


# kmanhat,ck3=k(X_train,5,manhat,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in kmanhat.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Manhatan distance', fontsize=12)


# # In[36]:


# #predict for the test data
# kmanhattest= predict(X_test,ck3,manhat)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in kmanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by k-means and Manhatan distance, test data', fontsize=12)


# # #### Fuzzy C-means

# # In[37]:


# feucli,cf1=fcm(X_train,5,eucli,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in feucli.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Euclidean distance', fontsize=12)


# # In[38]:


# #predict for the test data
# feuclitest= predict(X_test,cf1,eucli)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in feuclitest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Euclidean distance, test data', fontsize=11)


# # In[39]:


# fmahala,cf2=fcm(X_train,5,mahala,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in fmahala.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Mahalanobis distance', fontsize=12)


# # In[40]:


# #predict for the test data
# fmahalatest= predict(X_test,cf2,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in fmahalatest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Mahalanobis distance, test data', fontsize=11)


# # In[41]:


# fmanhat,cf3=fcm(X_train,5,manhat,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in fmanhat.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Manhatan distance', fontsize=12)


# # In[119]:


# #predict for the test data
# fmanhattest= predict(X_test,cf3,manhat)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in fmanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy c-means and Manhatan distance, test data', fontsize=11)


# # ### My algorithm

# # In[43]:


# peucli,cp1=per(X_train,5,eucli,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in peucli.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by custom k-means and Euclidean distance', fontsize=12)


# # In[44]:


# #predict for the test data
# peuclitest= predict(X_test,cp1,eucli)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in peuclitest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by custom k-means and Euclidean distance, test data', fontsize=11)


# # In[45]:


# pmahala,cp2=per(X_train,5,mahala,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in pmahala.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by custom k-means and Mahalanobis distance', fontsize=12)


# # In[46]:


# #predict for the test data
# pmahalatest= predict(X_test,cp2,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in pmahalatest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by custom k-means and Mahalanobis distance, test data', fontsize=11)


# # In[47]:


# pmanhat,cp3=per(X_train,5,manhat,100)#umapdf
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in pmanhat.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by custom k-means and Manhatan distance', fontsize=12)


# # In[48]:


# #predict for the test data
# pmanhattest= predict(X_test,cp3,manhat)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in pmanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by fuzzy custom k-means and Manhatan distance, test data', fontsize=11)


# # ## Mountain and substractive methods on $R^2$ and $R^3$ spaces
# # 
# # Before we can use these two methods, we must first find values for their respective parameters given the data at hand. In order to do that we will do some parameter sensibility analysis on them in UMAP 2D space and analize the graphs to choose the best parameters. 

# # In[124]:


# meucli,cm1=mount(normalize(umaptrain),5,eucli,100,sig=1.2,beta=1.2)
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in meucli.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain k-means and Euclidean distance', fontsize=12)


# # In[125]:


# meuclitest= predict(umaptest,cm1,eucli)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in meuclitest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain clustering and Euclidean distance, test data', fontsize=11)


# # After seeing this result, sigma and beta parameters for mountain will be set at 1.2 for the rest of the runs.
# # 
# # ra and rb will be set at 1.5

# # In[50]:


# mmahala,cm2=mount(normalize(umaptrain),5,mahala,100)
# mmahalatest= predict(umaptest,cm2,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in mmahalatest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain k-means and Mahalanobis distance, test data', fontsize=12)


# # In[134]:


# #smahala,cs2=sub(normalize(umaptrain),5,mahala,100)
# smahalatest= predict(umaptest,cs2,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in smahalatest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Mahalanobis distance, test data', fontsize=12)


# # In[51]:


# mmanhat,cm3=mount(normalize(umaptrain),5,manhat,100)
# mmanhattest= predict(umaptest,cm3,mahala)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in mmanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain k-means and Manhatan distance, test data', fontsize=12)


# # In[ ]:


# smanhat,cs3=sub(normalize(umaptrain),5,manhat,100)
# smanhattest= predict(umaptest,cs3,manhat)
# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in smanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Manhatan distance, test data', fontsize=12)


# # In[56]:


# tsne = TSNE(n_components=2)
# tsnetrainembed = tsne.fit_transform(X_train) 
# tsnetrain= pd.DataFrame()
# tsnetrain["x"] = tsnetrainembed[:,0]
# tsnetrain["y"] = tsnetrainembed[:,1]
# tsnetrain


# # In[147]:


# tsne = TSNE(n_components=2)
# tsnetestembed = tsne.fit_transform(X_test) 
# tsnetest= pd.DataFrame()
# tsnetest["x"] = tsnetestembed[:,0]
# tsnetest["y"] = tsnetestembed[:,1]
# tsnetest


# # In[57]:


# plt.scatter(
#     tsnetrainembed[:,0],
#     tsnetrainembed[:,1],
#     #c=[sns.color_palette()[x] for x in mmanhattest.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs dimension reduction with T-SNE', fontsize=12)


# # In[127]:


# meuclitsne,cm4=mount(tsnetrain,5,eucli,100)
# plt.scatter(
#     tsnetrainembed[:,0],
#     tsnetrainembed[:,1],
#     c=[sns.color_palette()[x] for x in meuclitsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain clustering and Euclidean distance, tsne reduction', fontsize=12)


# # In[59]:


# mmahalatsne,cm5=mount(tsnetrain,5,mahala,100)
# plt.scatter(
#     tsnetrainembed[:,0],
#     tsnetrainembed[:,1],
#     c=[sns.color_palette()[x] for x in mmahalatsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain clustering and Mahalanobis distance, tsne reduction', fontsize=12)


# # In[60]:


# mmanhattsne,cm6=mount(tsnetrain,5,manhat,100)
# plt.scatter(
#     tsnetrainembed[:,0],
#     tsnetrainembed[:,1],
#     c=[sns.color_palette()[x] for x in mmanhattsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain clustering and Manhatan distance, tsne reduction', fontsize=12)


# # In[66]:


# reducer = umap.UMAP(n_components=3)
# umap_train3D = reducer.fit_transform(X_train)


# # In[67]:


# umaptrain3D=pd.DataFrame(umap_train3D, columns=['x','y','z'])
# #umaptrain3D


# # In[68]:


# reducer = umap.UMAP(n_components=3)
# umap_test3D = reducer.fit_transform(X_test)


# # In[69]:


# umaptest3D=pd.DataFrame(umap_test3D, columns=['x','y','z'])
# #umaptest3D


# # In[81]:


# #meucli3D,cme3D=mount(normalize(umaptrain3D),5,eucli,100)
# #meuclitest= predict(umaptest3D,cme3D,eucli)
# ax = plt.axes(projection='3d')
# ax.scatter3D(
#     umap_test3D[:, 0],
#     umap_test3D[:, 1],
#     umap_test3D[:, 2],
#     c=[sns.color_palette()[x] for x in meuclitest.label]
#     )
# #plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by mountain k-means and Euclidean distance, test data in 3D', fontsize=12)


# # In[82]:


# mmahala3D,cmmahala3D=mount(normalize(umaptrain3D),5,mahala,100)
# mmahalatest= predict(umaptest3D,cmmahala3D,mahala)
# ax = plt.axes(projection='3d')
# ax.scatter3D(
#     umap_test3D[:, 0],
#     umap_test3D[:, 1],
#     umap_test3D[:, 2],
#     c=[sns.color_palette()[x] for x in mmahalatest.label]
#     )
# plt.title('Spotify songs clustered by mountain k-means and Mahalanobis distance, test data in 3D', fontsize=12)


# # In[83]:


# mmanhat3D,cmmanhat3D=mount(normalize(umaptrain3D),5,manhat,100)
# mmanhattest= predict(umaptest3D,cmmahala3D,manhat)
# ax = plt.axes(projection='3d')
# ax.scatter3D(
#     umap_test3D[:, 0],
#     umap_test3D[:, 1],
#     umap_test3D[:, 2],
#     c=[sns.color_palette()[x] for x in mmanhattest.label]
#     )
# plt.title('Spotify songs clustered by mountain k-means and Manhatan distance, test data in 3D', fontsize=12)


# # In[84]:


# X_train.to_excel('X_train.xlsx')
# X_test.to_excel('X_test.xlsx')
# X_Val.to_excel('Validation_Data.xlsx')


# # ### substractive
# # 

# # In[130]:


# cs1


# # In[129]:


# s,cs1=sub(umaptrain,5,eucli,100)
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in s.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Euclidean distance', fontsize=12)


# # In[131]:


# stsne,cs1tsne=sub(tsnetrain,5,eucli,100)
# plt.scatter(
#     umap_train[:, 0], #change this here
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in stsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Euclidean distance, tsne reduction', fontsize=12)


# # In[136]:


# plt.scatter(
#     umap_test[:, 0],
#     umap_test[:, 1],
#     c=[sns.color_palette()[x] for x in predict(umaptest,normalize(cs1),eucli).label]
#     )
# #predict(umaptest,normalize(cs1),eucli)


# # In[137]:


# smanhat,cs2=sub(umaptrain,5,manhat,100)
# plt.scatter(
#     umap_train[:, 0],
#     umap_train[:, 1],
#     c=[sns.color_palette()[x] for x in smanhat.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Manhatan distance', fontsize=12)


# # In[144]:


# #smanhattsne,cs2tsne=sub(tsnetrain,5,manhat,100)
# plt.scatter(
#     tsnetrainembed[:,0],
#     tsnetrainembed[:,1],
#     c=[sns.color_palette()[x] for x in smanhattsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Manhatan distance, tsne reduction', fontsize=12)


# # In[154]:


# #cs2tsne
# stesttsne= predict(tsnetest,normalize(cs2tsne),eucli)
# plt.scatter(
#     tsnetestembed[:,0],
#     tsnetestembed[:,1],
#     c=[sns.color_palette()[x] for x in stesttsne.label]
#     )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Spotify songs clustered by substractive clustering and Manhatan distance, tsne reduction, test data', fontsize=10)

