
import random

import seaborn as sns

#Author - Shekhar Dewan


sns.set()
from PIL import Image
MAX_ITERATIONS = 500

def kmeans(features, k):


    centroids = random.sample(features,k)

    iterations = 0
    oldCentroids = []

    while not shouldStop(oldCentroids, centroids, iterations):
        oldCentroids = centroids
        iterations += 1

        labels = getLabels(features, centroids)

        centroids = getCentroids(features, labels, k)

        #sys.stdout.write("\r"+str(iterations))
        #sys.stdout.flush()
    #print("\rTook", iterations, "iterations")

    return labels, centroids

def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True
    return oldCentroids == centroids

def getLabels(dataSet, centroids):
    #
    assigments = []
    for data in dataSet:
        min_dist = None
        cluster = 0
        for i, cent in enumerate(centroids):
            dist = 0
            dist += (data[0]-cent[0])**2
            dist += (data[1]-cent[1])**2
            dist += (data[2]-cent[2])**2
            if(min_dist is None or dist < min_dist):
                min_dist = dist
                cluster = i
        assigments.append(cluster)
    #plt.hist(assigments)
    return assigments

def getCentroids(dataSet, labels, k):
    mean_value = []
    for i in range(k):
        x_total = 0
        y_total = 0
        l_total = 0
        x_avg = 0
        y_avg = 0
        l_avg = 0
        num_in_cluster = 0
        for j in range(len(dataSet)):
            if(labels[j] == i):
               x_total += dataSet[j][0]
               y_total += dataSet[j][1]
               l_total += dataSet[j][2]
               num_in_cluster += 1
        if(num_in_cluster > 0):
            x_avg = x_total/num_in_cluster
            y_avg = y_total/num_in_cluster
            l_avg = l_total/num_in_cluster
        mean_value.append((x_avg,y_avg,l_avg))
    return mean_value


image = Image.open("test4.jpg")
luminance = image.convert("L")
#luminance.show() - does convert to greyscale

width, height = image.size

lumi_tuples = []
for i in range (0,width):
    for j in range (0,height):
        lumi_tuples.append((i,j,luminance.getpixel((i,j))))

for i in range(2,10):
    numClusters = i
    k_labeled,k_clustered = kmeans(lumi_tuples,numClusters)
    print(k_clustered)

    clusteredImg = Image.new("L",(width,height))
    l = 0
    for i in range (0,width):
        for j in range (0,height):
            #k = i*width + j
            #color = k_labeled[k]
            color = k_labeled[l]
            l += 1
            clusteredImg.putpixel((i, j), (255//(numClusters-1)*color))

    clusteredImg.show()
    clusteredImg.save("test4_" + str(numClusters) + "_clusters.jpg")
