import numpy as np
#from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans as skmeans
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import nltk, itertools, string, unicodedata, re
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import scipy.sparse as sp
from matplotlib import transforms


#Tabla de elementos de puntuación
punctuation_table = str.maketrans({key: None for key in string.punctuation+"¿¡?!"})
#Stop words sin acentos
stop_words = [unicodedata.normalize('NFKD', stw).encode('ASCII', 'ignore').decode()
              for stw in nltk.corpus.stopwords.words('spanish')]

#distancia euclideana
def euclidiana(x,y):
    m=x-y
    return np.sqrt(np.sum(m*m))

#distancia coseno
def coseno(x,y):
    sim=np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))
    # si los vectores ya están normalizados se podría utilizar la siguente linea
    #dist=np.dot(x, y.T)[0,0]
    return 1-sim

# Función para graficar clusters en dos y tres dimensiones
def plotClusters(data,labels,centroids={},f="",centroids_txt_labels={}):
    fig=plt.figure(figsize=(6, 6))
    sbox = dict(boxstyle='round', facecolor='white', alpha=0.4)
    d=len(data[1])
    if d==3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    K=np.unique(labels)
    color_map=iter(cm.viridis(np.linspace(0,1,len(K))))
    for k in K:
        D=data[np.where(labels==k)]
        x,y=D[:,0],D[:,1]
        cl=next(color_map)
        if d==3:
            z=D[:,2]
            ax.scatter(x,y,z, color=cl,s=32)
        else:
            ax.scatter(x,y, color=cl,s=32)
        if len(centroids):
            txt_label=centroids_txt_labels and str(centroids_txt_labels[k]) or str(k)
            if len(centroids[k])==3:
                xc,yc,zc=centroids[k]
                ax.text(xc,yc,zc,txt_label,bbox=sbox,fontsize=14)
            else:
                xc,yc=centroids[k]
                ax.text(xc,yc,txt_label,bbox=sbox,fontsize=14)
    if len(data[0])==3:
        ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    if f:
        fig.savefig(f)

# Transformar datos N>3 dimensionales a 2 o tres dimensiones usando PCA                  
def plotPCA(data, labels=[]):
    pca = TruncatedSVD(2,random_state=100)
    pca.fit(data)
    X=pca.transform(data)
    n=len(X)
    if len(labels)==0:
        labels=[f"doc_{i}" for i in range(n)]
    origin2d=[0 for x in range(n)]
    origin3d=[0],[0],[0]
    xM,xm=np.max(X[:,0]),np.min(X[:,0])
    yM,ym=np.max(X[:,1]),np.min(X[:,1])
    sbox = dict(boxstyle='round', facecolor='white', alpha=0.4)
    for x,l in zip(X,labels):
         plt.text(x[0],x[1],l,bbox=sbox)
    plt.quiver(origin2d,origin2d, X[:,0],X[:,1],angles='xy',scale=1,
                   scale_units='xy', color='skyblue') 
    plt.xlim(xm-0.1,xM+0.1)
    plt.ylim(ym-0.1,yM+0.1)

#Método bruto para calcular  atriz de cocurrencia de palabrar 
def cocurrency_matrix(sentences):
    voc=list(set(list(itertools.chain.from_iterable(sentences))))
    voc.sort()
    V=len(voc)
    matrix=pd.DataFrame(index=voc,columns=voc,data=np.zeros((V,V)))
    for word in voc:
        for sentence in sentences:
            if word in sentence:
                for col in set(sentence):
                    matrix[word][col]+=1
                    #if col!=word:
                    #    matrix[col][word]=matrix[col][word]+1
    return np.array(voc),matrix

#Para tokenizar una lista de textos                            
def tokenize_sentences(texts):
    tokenized_texts=[preprocess(txt) for txt in texts]
    return np.array(tokenized_texts,dtype=object)

# Preprocesamiento simple
def preprocess(sentence):
     st=sentence.lower()
     st=re.sub(r"http\S+", "", st)
     st=st.translate(punctuation_table)   
     st=unicodedata.normalize('NFKD', st).encode('ASCII', 'ignore').decode()
     tokens=[word for word in word_tokenize(st) if word not in stop_words]
     return tokens

def my_tokenizer(txt):
    return preprocess(txt)


    

    
                                
    

         
