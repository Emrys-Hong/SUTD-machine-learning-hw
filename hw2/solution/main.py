import numpy as np 
import matplotlib.pyplot as plt 

from kmns import kmns_algo, cluster_d

# load image data
data = np.loadtxt('hw2-image.txt')
data = np.asarray(data, dtype=np.float)

# initial centroids
n_centroids = 8
centroids = [[255, 255, 255],
             [255, 0, 0],
             [128, 0, 0],
             [0, 255, 0],
             [0, 128, 0],
             [0, 0, 255],
             [0, 0, 128],
             [0, 0, 0]]
centroids = np.asarray(centroids, dtype=np.float) 

# perform k-means algo 
num_iter, data_index, new_centroids, error_list = kmns_algo(data, n_centroids, centroids)
#'''
print(num_iter, new_centroids)
plt.figure()
plt.plot(range(num_iter), error_list )
plt.xlabel('Num. of Iter.')
plt.ylabel('Sum of Error')
plt.savefig('./sum_error_vs_iter.png')
#plt.show()
#'''

# visualize the reuslt by replacing the pixels
size = [516,407]
'''
centroids = [[241.2296146 , 238.62515213, 233.86288032], 
             [194.41158657, 136.33311389,  90.94364714], 
             [136.2655563 ,  61.08973066,  10.10385457], 
             [np.float("inf"), np.float("inf"), np.float("inf")], 
             [157.29173273,  97.59397508,  51.43329558], 
             [np.float("inf"), np.float("inf"), np.float("inf")], 
             [78.92743714, 37.10828688, 13.07070482], 
             [25.97800232, 23.23575423, 23.60599063]]
'''
centroids = new_centroids

centroids = np.asarray(centroids, dtype=np.float)

# find closest centroids and get cluster index 
data_index = cluster_d(data, n_centroids, centroids)

new_data = np.zeros((size[0],size[1])) # image size
new_data = []
for row in range(size[0]):   
    new_row = []
    for col in range(size[1]):
        i = row * size[1] + col
        idx = data_index[i]
        new_pixel = centroids[idx].astype(np.uint8)
        new_row.append(new_pixel)
    new_row = np.asarray(new_row)
    new_data.append(new_row)

new_data=np.asarray(new_data)
 
plt.figure()
plt.imshow(new_data, vmin=0,vmax=255)
plt.savefig('./visualize_after_replacing.png')


