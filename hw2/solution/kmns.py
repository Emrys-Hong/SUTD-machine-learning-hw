import numpy as np 

def dist_cal(x,y): # calculate squared distance for argmin
    # distance between single data point with all centroids
    temp = np.square(x-y)
    dist=temp.sum(axis=1)
    distance = np.asarray(dist, dtype=np.float)
    return np.sqrt(distance)


# find closest centroids
def cluster_d(data, n_centroids, centroids):
    data_index = []
    for i in range(len(data)):
        distance = dist_cal(data[i], centroids)
         
        index = np.argmin(distance)
        data_index.append(index)
    
    return data_index

# calcualte distance between x and y
def se_dis_cal(x,y): 
    error = np.square(x-y)
    error = np.sqrt(error.sum())
    return np.asarray(error, dtype=np.float)

# calculate new centroids
def centroids_cal(data_index, data, n_centroids):
    new_centroids = []
    data_index = np.asarray(data_index)
    len_select=[]
    for i in range(n_centroids):
        select = data_index == i
        select_data = data[select]
        len_select.append(len(select_data))
        #print('select',select_data)
        if len(select_data) == 0:
            new_centroid = np.array([np.float("inf"),np.float("inf"),np.float("inf")])
        else:
            new_centroid = select_data.mean(axis=0)
        new_centroids.append(new_centroid)
    print(len_select)
    return new_centroids

# calculate the sum error
def error_cal(data_index, data, n_centroids, new_centroids):
    num=len(data_index)
    sum_error=0
    for k in range(num):
        idx = data_index[k]
        error = se_dis_cal(data[k],new_centroids[idx])
        sum_error = sum_error + error.sum()
    return sum_error

# k means algorithm                 
def kmns_algo(data, n_centroids, centroids):
    max_iter = 100
    iter = 0
    error_list = []
    old_error = 0
    for i in range(max_iter):
        data_index = cluster_d(data, n_centroids, centroids)
        centroids = centroids_cal(data_index, data, n_centroids)
        sum_error = error_cal(data_index, data, n_centroids, centroids)
        iter = iter +1
        print('iter:', iter, 'sum_error:', sum_error)
        #print(centroids)
        error_list.append(sum_error)
        if old_error == sum_error:  # stop criterion
            break
        old_error = sum_error
    return iter, data_index, centroids, error_list