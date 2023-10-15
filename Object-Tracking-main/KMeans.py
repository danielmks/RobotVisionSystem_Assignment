import numpy as np
class KmeansSegmentation:

    def segmentation_grey(self, image, k = 2):
        """Performs segmentation of an grey level input image using KMeans algorithm, using the intensity of the pixels as features
        takes as input:
        image: a grey scale image
        return an segemented image
        -----------------------------------------------------
        Sample implementation for K-means
        1. Initialize cluster centers
        2. Assign pixels to cluster based on (intensity) proximity to cluster centers
        3. While new cluster centers have moved:
            1. compute new cluster centers based on the pixels
            2. Assign pixels to cluster based on the proximity to new cluster centers

        """
        #assigning cluster centroids clusters
        centroids = []
        clusters=[]

        i=1
        # Initializes k number of centroids for the clustering making sure no cluster centroids are same

        while(len(centroids)!=k):
            cent = image[np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
            if(len(centroids)>=1):
                if(cent not in centroids):
                    centroids.append(cent)
            else:
                centroids.append(cent)
        print("Initial centroids are {}".format(centroids))

        # Initializing k clusters
        for m in range(0, k):
            cluster=[]
            clusters.append(cluster)

        # Calling k means which returns the clusters with pixels
        clusters = self.kmeans(clusters, image, centroids, k,"grey")
        new_centroids=self.calculate_new_centroids(clusters,k,"grey")

        # clustering and finding new centroids till convergence is reached
        while(not(np.array_equal(new_centroids,centroids))) and i<=15:
            centroids=new_centroids
            clusters=self.kmeans(clusters,image,centroids,k,"grey")
            new_centroids = self.calculate_new_centroids(clusters, k,"grey")
            i=i+1
        print("Convergence reached")

        image=self.assignPixels(clusters,image,k,"grey")
        return image

    def findMinIndex(self,pixel, centroids,kind):
        if(kind=="grey"):
            d = []
            for i in range(0, len(centroids)):
                d1 = abs(int(pixel) - centroids[i])
                d.append(d1)
            minIndex = d.index(min(d))
        elif(kind=="rgb"):
            dist=[]
            for i in range(0, len(centroids)):
                d2 = np.sqrt(int((centroids[i][0] - pixel[0])) ** 2 + int((centroids[i][1] - pixel[1])) ** 2 + int(
                    (centroids[i][2] - pixel[2])) ** 2)
                dist.append(d2)
                minIndex=dist.index(min(dist))

        return minIndex

    def assignPixels(self,clusters,image,k,kind):
        if(kind=="grey"):
            cluster_centroids=[]
            for i in range(0, k):
                cent = np.nanmean(clusters[i])
                cluster_centroids.append(cent)

            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    Value = round(cluster_centroids[self.findMinIndex(image[x,y], cluster_centroids,kind)])
                    image[x, y] = Value

        elif(kind=="rgb"):
            cluster_centroids = []
            keys = sorted(clusters.keys())
            for k in keys:
                n_mean = np.mean(clusters[k], axis=0)
                cent_new = (int(n_mean[0]), int(n_mean[1]), int(n_mean[2]))
                cluster_centroids.append(cent_new)

            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    Value = cluster_centroids[self.findMinIndex(image[x,y], cluster_centroids,kind)]
                    image[x, y] = Value
        return image

    def kmeans(self, clusters, image, centroids, k, kind):

        def add_cluster(minIndex, pixel):
            try:
                clusters[minIndex].append(pixel)
            except KeyError:
                clusters[minIndex] = [pixel]

        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                pixel = image[x, y].tolist()
                minIndex = self.findMinIndex(pixel, centroids,kind)
                add_cluster(minIndex, pixel)

        return clusters

    def calculate_new_centroids(self,clusters,k,kind):
        if(kind=="grey"):
            new_centroids=[]
            for i in range(0, k):
                cent = np.nanmean(clusters[i])
                new_centroids.append(round(cent))

        elif(kind=="rgb"):
            new_centroids = []
            keys = sorted(clusters.keys())
            for k in keys:
                n_mean = np.mean(clusters[k], axis=0)
                cent_new = (int(n_mean[0]), int(n_mean[1]), int(n_mean[2]))
                new_centroids.append(cent_new)

        return new_centroids

    def segmentation_rgb(self, image, k = 2):
        """Performs segmentation of a color input image using KMeans algorithm, using the intensity of the pixels (R, G, B)
        as features
        takes as input:
        image: a color image
        return an segemented image"""
        centroids = []
        clusters = {}

        i=1
        # Initializes k number of centroids for the clustering making sure that no centroids chosen randomly are same
        while(len(centroids)!=k):
            cent = image[np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
            if(len(centroids)>=1):
                if(cent.tolist() not in centroids):
                    centroids.append(cent.tolist())
            else:
                centroids.append(cent.tolist())
        print("Initial centroids are {}".format(centroids))

        clusters = self.kmeans(clusters, image, centroids, k,"rgb")
        new_centroids = self.calculate_new_centroids(clusters, k,"rgb")

        # clustering and finding new centroids till convergence is reached
        while(not(np.array_equal(new_centroids, centroids))) and i<=15:
            centroids = new_centroids
            clusters = self.kmeans(clusters, image, centroids, k,"rgb")
            new_centroids = self.calculate_new_centroids(clusters, k,"rgb")
            i=i+1
        else:
            print("Convergence reached!")
        # printing the end clusters
        # for i in range(0, k):
        #     print("Printing cluster which is {}".format(i))
        #     print(clusters[i])
        image = self.assignPixels(clusters, image, k,"rgb")
        return image