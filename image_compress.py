from matplotlib import pyplot as io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

if len(sys.argv)<4:
    print("Enter proper agruments")
    print("<image_name> <#clusters> <#iterations> <file_name to save (optional) >")
    sys.exit()

class KMeans:
    def __init__(self,k):
        self.k=k

    def findClosestCentroids(self,x,centroids):
        m,n=x.shape
        k=len(centroids)
        idx=np.zeros((m,1))
        for i in range(m):
            displaydata=np.zeros((1,k))
            for j in range(k):
                displaydata[:,j]=np.sqrt(np.sum(np.power((x[i,:]-centroids[j,:]),2)))
            idx[i]=np.argmin(displaydata)+1
        return idx


    def computeCentroids(self,x,idx):
        m,n=x.shape
        centroids=np.zeros((self.k,n))
        count=np.zeros((self.k,1))
        for i in range(m):
            index=int(idx[i]-1)
            centroids[index,:]+=x[i,:]
            count[index]+=1
        return centroids/count

    def plot_KMeans(self,x,idx,centroid,num_iter):
        m,n=x.shape
        fig,ax=plt.subplots(nrows=num_iter,ncols=1,figsize=(10,6))
        for i in range(num_iter):
            color='rbg'
            for k in range(self.k+1):
                grp=(idx==k).reshape(m,1)
                ax[i].scatter(x[grp[:,0],0],x[grp[:,0],1],c=color[k-1],s=15)

            ax[i].scatter(centroid[:,0],centroid[:,1],color='Black',s=150,marker='x')
            idx=self.findClosestCentroids(x,centroid)
            centroid=self.computeCentroids(x,idx)
            ax[i].set_title('iteration:'+str(i))


    def init_random_centroid(self,x):
        m,n=x.shape
        centroid=np.zeros((self.k,n))
        for i in range(self.k):
            centroid[i]=x[np.random.randint(0,m+1),:]
        return centroid

    def runKMeans(self,x,centroids,num_iter):
        idx=self.findClosestCentroids(x,centroids)
        for i in range(num_iter):
            centroids=self.computeCentroids(x,idx)
            idx=self.findClosestCentroids(x,centroids)

        return centroids,idx






# imagefilename='Penguins.jpg'
# imagefilename='Koala.jpg'
imagefilename=sys.argv[1]

img1 = io.imread(imagefilename) #image is saved as rows * columns * 3 array

# print(img1)
# print(img1.shape)
# plt.imshow(img1)
# plt.show()

X=(img1/255).reshape(img1.shape[0]*img1.shape[1],3)

def compressImage(X,k,num_iter):
    kmeans=KMeans(k)
    initial_centroid=kmeans.init_random_centroid(X)
    compressed_centroid,compressed_idx=kmeans.runKMeans(X,initial_centroid,num_iter)

    X_compressed=X.copy()
    for i in range(1,k+1):
        X_compressed[(compressed_idx==i).ravel(),:] = compressed_centroid[i-1]
    return X_compressed


def plot_img(original,compressed):
    fig,ax=plt.subplots(1,2)
    img1=ax[0].imshow(original)
    img2=ax[1].imshow(compressed)
    for i in range(2):
        title=['original','compressed']
        ax[i].set_title(title[i])
    plt.show()


X_compressed=compressImage(X,int(sys.argv[2]),int(sys.argv[3]))  # call the function

X_compressed=X_compressed.reshape(img1.shape[0],img1.shape[1],3)
X=X.reshape(img1.shape[0],img1.shape[1],3)
simg=Image.fromarray((X_compressed*255).astype(np.uint8))

if len(sys.argv)==4:
    save_file=sys.argv[2]+sys.argv[1]    # #clusters_imgname
else: 
    save_file=sys.argv[4]
# save_file=sys.argv[4]
# simg.save('20cluster.jpg')
# simg.save('5cluster-p.jpg')
simg.save(save_file)

# plot_img(X,X_compressed)