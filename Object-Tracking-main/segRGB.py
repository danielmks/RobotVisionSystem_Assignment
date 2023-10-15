import numpy as np


class KmeansSeg:

    def findMinIndex(self, pixel, centroids):#pixel에서 가장 가까운 cluster 찾기
        dist = []
        for i in range(0, len(centroids)):
            d = np.sqrt(int((centroids[i][0] - pixel[0])) ** 2 + int((centroids[i][1] - pixel[1])) ** 2 + int(
                (centroids[i][2] - pixel[2])) ** 2)                                                      #pixel의 RGB값과 centroid을 거리를 계산
            dist.append(d)
            minIndex = dist.index(min(dist))                #pixel가 가장 가까운 cluster의 index 찾기

        return minIndex

    def assignPixels(self, clusters, image, k):  #cluster의 RGB값을 통해 output image의 RGB값을 계산
        cluster_centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            n_mean = np.mean(clusters[k], axis=0)
            cent_new = (int(n_mean[0]), int(n_mean[1]), int(n_mean[2]))
            cluster_centroids.append(cent_new)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                Value = cluster_centroids[self.findMinIndex(image[x, y], cluster_centroids)]
                image[x, y] = Value

        return image

    def kmeans(self, clusters, image, centroids, k): #각각의 pixel들을 가장 가까운 centroids의 clusters에 assgin

        def add_cluster(minIndex, pixel):
            try:
                clusters[minIndex].append(pixel)
            except KeyError:
                clusters[minIndex] = [pixel]

        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                pixel = image[x, y].tolist()
                minIndex = self.findMinIndex(pixel, centroids)
                add_cluster(minIndex, pixel)        #가장 가까운 cluster에 pixel의 RGB값을 저장

        return clusters

    def get_new_centroids(self, clusters, k): # cluster에 clustering된 RGB값의 mean을 통해 새로운 centroid를 계산
        new_centroids = []
        keys = sorted(clusters.keys())
        for k in keys:
            n_mean = np.mean(clusters[k], axis=0)
            cent_new = (int(n_mean[0]), int(n_mean[1]), int(n_mean[2]))
            new_centroids.append(cent_new)

        return new_centroids

    def segmentation(self, image, k=2):
        centroids = []
        clusters = {}
        i = 1

        while (len(centroids) != k):
            cent = image[np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])]
            if (len(centroids) >= 1):
                if (cent.tolist() not in centroids):
                    centroids.append(cent.tolist())
            else:
                centroids.append(cent.tolist())      #centroid의 초기값을 랜덤으로 정해준다 centroids list에 추가
        print("Initial centroids {}".format(centroids))

        clusters = self.kmeans(clusters, image, centroids, k)   #kmean clustering 실행
        new_centroids = self.get_new_centroids(clusters, k)     #새로운 centroids 할당

        while (not (np.array_equal(new_centroids, centroids))) and i <= 15: # 수렴할 때까지 반복(max치 15)
            centroids = new_centroids
            clusters = self.kmeans(clusters, image, centroids, k)
            new_centroids = self.get_new_centroids(clusters, k)
            i = i + 1
        else:
            print("END")

        image = self.assignPixels(clusters, image, k)

        return image
