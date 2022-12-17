import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import kmeans_plusplus

# np.set_printoptions(threshold=sys.maxsize)

#1.1(a) Load .mat file and convert into np array
train_data = scipy.io.loadmat('sarcos_inv.mat')
test_data = scipy.io.loadmat('sarcos_inv_test.mat')

train_data = train_data['sarcos_inv']
test_data = test_data['sarcos_inv_test']

#1.1(b) Split into 6 arrays
num_of_rows = np.shape(train_data)[0]
idx = np.random.choice(num_of_rows, size = int(num_of_rows * 0.8), replace = False)
xs_train = train_data[idx, :]
ys_train = xs_train[:, 21:22]
xs_train = xs_train[:, 0:21]
xs_valid = np.delete(train_data, idx, axis = 0)
ys_valid = xs_valid[:, 21:22]
xs_valid = xs_valid[:, 0:21]
xs_test = test_data[:, 0:21]
ys_test = test_data[:, 21:22]

assert xs_train.shape == (35587, 21), "xs_train should contain 35587 21-dimensional data points"
assert ys_train.shape == (35587, 1), "ys_train should contain 35587 1-dimensional data points"
assert xs_valid.shape == (8897, 21), "xs_valid should contain 8897 21-dimensional data points"
assert ys_valid.shape == (8897, 1), "ys_valid should contain 8897 1-dimensional data points"
assert xs_test.shape == (4449, 21), "xs_test should contain 4449 21-dimensional data points"
assert ys_test.shape == (4449, 1), "ys_test should contain 4449 1-dimensional data points"

# 1.2(a) Find variance
def my_variance(xs: np.ndarray) -> np.ndarray:
    var = np.var(xs)
    return var
    raise NotImplementedError()

assert np.isclose(my_variance(np.array([1, 1, 1])), 0), "Variance of this vector should be 0"
assert np.isclose(my_variance(np.array([1, 2, 3, 4, 5])), 2), "Variance of this vector should be 2"

# 1.2(b) Find MSE for 1-D array
def my_mse(z1: np.ndarray, z2: np.ndarray):
    """ Computes the Mean Squared Error (MSE)
    
    Args:
        z1: A 1D numpy array (usually the predictions).
        z2: Another 1D numpy array.
    
    Returns
        The MSE of the given data.
    """
    # YOUR CODE HERE
    mse = 0
    idx = np.shape(z1)[0]
    for x in range(0, idx):
        mse = mse + pow(z2[x] - z1[x], 2)
    
    mse = mse / idx
    
    return mse
    raise NotImplementedError()

assert np.isclose(my_mse(np.array([3.0]), np.array([4.0])), 1), "The MSE between 3 and 4 should be 1"
assert np.isclose(my_mse(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])), 0), "MSE should be 0 for identical z vectors"

# 1.3 Standardization
# idx = np.shape(xs_train)[0]
# xs_train_std = np.ones((idx, 21))
# for x in range(0, idx):
#     mean = np.mean(xs_train[x, :])
#     xs_train_std[x] = xs_train[x, :] - mean
#     std = np.std(xs_train[x, :])
#     xs_train_std[x] = xs_train_std[x, :] / std
# print(np.var(xs_train_std, axis = 0))

idx = np.shape(xs_train)[1]
xs_train_std = np.ones((np.shape(xs_train)[0], np.shape(xs_train)[1]))
for x in range(0, idx):
    mean = np.mean(xs_train[:, x])
    xs_train_std[:, x] = xs_train[:, x] - mean
    std = np.std(xs_train[: ,x])
    xs_train_std[:, x] = xs_train_std[:, x] / std

idx = np.shape(xs_valid)[1]
xs_valid_std = np.ones((np.shape(xs_valid)[0], np.shape(xs_valid)[1]))
for x in range(0, idx):
    mean = np.mean(xs_valid[:, x])
    xs_valid_std[:, x] = xs_valid[:, x] - mean
    std = np.std(xs_valid[: ,x])
    xs_valid_std[:, x] = xs_valid_std[:, x] / std

idx = np.shape(xs_test)[1]
xs_test_std = np.ones((np.shape(xs_test)[0], np.shape(xs_test)[1]))
for x in range(0, idx):
    mean = np.mean(xs_test[:, x])
    xs_test_std[:, x] = xs_test[:, x] - mean
    std = np.std(xs_test[: ,x])
    xs_test_std[:, x] = xs_test_std[:, x] / std

idx = np.shape(ys_train)[0]
ys_train_std = np.ones((idx, 1))
mean = np.mean(ys_train)
ys_train_std[:] = ys_train[:] - mean

idx = np.shape(ys_valid)[0]
ys_valid_std = np.ones((idx, 1))
mean = np.mean(ys_valid)
ys_valid_std[:] = ys_valid[:] - mean

idx = np.shape(ys_test)[0]
ys_test_std = np.ones((idx, 1))
mean = np.mean(ys_test)
ys_test_std[:] = ys_test[:] - mean

assert xs_train_std.shape == xs_train.shape, "Normalizing is not supposed to change the shape of your data"
assert ys_train_std.shape == ys_train.shape, "Normalizing is not supposed to change the shape of your data"
assert xs_valid_std.shape == xs_valid.shape, "Normalizing is not supposed to change the shape of your data"
assert ys_valid_std.shape == ys_valid.shape, "Normalizing is not supposed to change the shape of your data"
assert xs_test_std.shape == xs_test.shape, "Normalizing is not supposed to change the shape of your data"
assert ys_test_std.shape == ys_test.shape, "Normalizing is not supposed to change the shape of your data"
assert np.isclose(np.mean(xs_train_std), 0, atol=0.005), "Training inputs mean should be 0"
assert np.isclose(np.mean(ys_train_std), 0, atol=0.005), "Training outputs mean should be 0"
assert np.allclose(np.var(xs_train_std, axis=0), 1, atol=0.005), "Training inputs variance should be 1"
assert np.isclose(np.mean(xs_valid_std), 0, atol=0.005), "Training inputs mean should be 0"
assert np.isclose(np.mean(ys_valid_std), 0, atol=0.005), "Training outputs mean should be 0"
assert np.allclose(np.var(xs_valid_std, axis=0), 1, atol=0.005), "Training inputs variance should be 1"
assert np.isclose(np.mean(xs_test_std), 0, atol=0.005), "Training inputs mean should be 0"
assert np.isclose(np.mean(ys_test_std), 0, atol=0.005), "Training outputs mean should be 0"
assert np.allclose(np.var(xs_test_std, axis=0), 1, atol=0.005), "Training inputs variance should be 1"

# 1.4 Determine weight vector
def my_linear_regression(phi: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """ Computes the weights of a linear regression that fits the given data.
    
    Notes:
        You may use np.linalg.solve to solve a system of linear equations.
    
    Args:
        phi: Input feature matrix of shape (N, D) containing N samples of dimension D.
        ys: Target outputs of shape (N, 1) containing N 1-dimensional samples.
        
    Returns:
        A numpy array containing the regressed weights of shape (D, 1), containing one weight for each input dimension.
    """
    # YOUR CODE HERE
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.transpose(phi)), ys)
    return w
    raise NotImplementedError()

_my_weights = my_linear_regression(xs_train_std, ys_train_std)
assert _my_weights.shape == (21, 1), "Weights should have shape (D, 1)."
_my_y_valid_pred = xs_valid_std @ _my_weights
_my_mse = my_mse(ys_valid_std, _my_y_valid_pred)
print(f"Your MSE should be roughly 31 and it is {_my_mse}.")

# 1.5
def my_quadratic_features(xs: np.ndarray) -> np.ndarray:
    """ Generates polynomial features up to degree 2 from given data.
    
    The quadratic features should include monomials (i.e., x_i, x_i**2 etc)
    and interaction terms (x_1*x_2 etc), but no repetitions (i.e. NOT both x_1*x_2 and x_2*x_1).
    You should include a bias term.
    The order of the samples should not be changed through the transformation.
    
    Args:
        xs: A 2D numpy array of shape (N, D) containing N samples of dimension D.
    
    Returns:
        An (N, M) numpy array containing the transformed input.
    """
    # YOUR CODE HERE
    N = xs.shape[0]
    D = xs.shape[1]
    out = []
    num = 1 + D + D * (D + 1) / 2
    num = int(num)

    for i in range(N):
        x_arr = np.insert(xs[i], 0, [1])
        for j, n in enumerate(x_arr.tolist()):
            for k, m in enumerate(x_arr.tolist()):
                if j <= k:
                    out = np.append(out, [n*m])
    
    out = out.reshape(N, num)
    
    return out
    raise NotImplementedError()

print(np.shape(my_quadratic_features(np.array([[0, 1, 2,3,4,5,6]]))))
assert my_quadratic_features(np.array([[0, 1]])).shape == (1, 6), "For 2D data, your function should produce 6D quadratic features."
assert my_quadratic_features(np.array([[0, 1], [2, 3]])).shape == (2, 6), "Your function should produce 6D quadratic features for every data point."

# 2.1
def my_kmeans(xs: np.ndarray, init_centers: np.ndarray, n_iter: int):
    """ Runs the K-Means algorithm from a given initialization
    
    Args:
        xs: A 2D numpy array of shape (N, D) containing N samples of dimension D
        init_centers: A 2D numpy array of shape (K, D) containing the K initial cluster centers of dimension D.
        n_iter: The number of iterations for the K-Means algorithm.
    
    Returns:
        A (K, D) numpy array containing the final cluster centers.
    """
    
    # YOUR CODE HERE
    r = np.zeros((np.shape(xs)[0], np.shape(init_centers)[0]))
    A = init_centers

    for iter in range(0, n_iter):
        for i in range(0, np.shape(xs)[0]):
            min_dis = np.linalg.norm(xs[i] - A[0])
            min_idx = 0
            for j in range(0, np.shape(A)[0]):
                if np.linalg.norm(xs[i] - A[j]) < min_dis:
                    min_idx = j
            # print(min_idx)
            for k in range(0, np.shape(r)[1]):
                # print(np.shape(A))
                # print(np.shape(r))
                # print(k)
                # print(min_idx)
                # print(' ')
                if k != min_idx:
                    r[i, k] = 0
                else:
                    r[i, k] = 1
            # print(r)
            # print(' ')

        for i in range(0, np.shape(A)[0]):
            num_of_pt_sum = 0
            center_sum = np.zeros((1, np.shape(A)[1]))
            for j in range(0, np.shape(r)[0]):
                if r[j, i] == 1:
                    num_of_pt_sum += 1
                    center_sum += xs[j]
            center_sum = center_sum / num_of_pt_sum
            # print(num_of_pt_sum)
            A[i] = center_sum
        # print(A)
    
    return A
    raise NotImplementedError()

assert my_kmeans(
    np.array([[0.92222276, 0.65417794, 0.81171083], [0.32436396, 0.43398054, 0.06203346], [0.66190191, 0.51464817, 0.53506438], [0.52361743, 0.82799732, 0.99989914]]),
    np.array([[0.46606325, 0.30170084, 0.3454716], [0.09386854, 0.6876939 , 0.89328422]]),
    3
).shape == (2, 3), "Final cluster centers must have the same shape as the initial cluster centers"
assert np.allclose(my_kmeans(np.random.rand(10, 4), __initial_centers := np.random.rand(3, 4), 0), __initial_centers), "For 0 iterations, the final cluster centers must be identical to the initial cluster centers"

# 2.2
# xs_cluster_test = ...
# # YOUR CODE HERE
# gaussian_1 = np.random.multivariate_normal([-2, 2], 0.2 * np.eye(2), 30)
# gaussian_2 = np.random.multivariate_normal([-2, -2], 0.2 * np.eye(2), 20)
# gaussian_3 = np.random.multivariate_normal([2, -2], 0.5 * np.eye(2), 40)
# gaussian_4 = np.random.multivariate_normal([2, 2], 0.5 * np.eye(2), 10)
# xs_cluster_test = np.concatenate((gaussian_1, gaussian_2, gaussian_3, gaussian_4), axis = 0)
# plt.scatter(xs_cluster_test[:, 0], xs_cluster_test[:, 1])
# plt.show()
# raise NotImplementedError()

# np.random.seed(1234)
idx = np.random.choice(4, 100, p = [0.3, 0.2, 0.4, 0.1])
xs_cluster_test = np.zeros((100,2))

means = np.array([[-2,2], [-2,-2], [2,-2], [2,2]])
cov = np.array([[[0.2,0],[0,0.2]],
                [[0.2,0],[0,0.2]],
                [[0.5,0],[0,0.5]],
                [[0.5,0],[0,0.5]]])

def Guassian_distribution(m):
    if m == 0:
        return np.random.multivariate_normal([-2,2], [[0.2,0], [0,0.2]], 1)
    elif m == 1:
        return np.random.multivariate_normal([-2,-2], [[0.2,0], [0,0.2]], 1)
    elif m == 2:
         return np.random.multivariate_normal([2,-2], [[0.5,0], [0,0.5]], 1)
    else :
        return np.random.multivariate_normal([2,2], [[0.2,0], [0,0.5]], 1)

for index, i in enumerate(idx):
    xs_cluster_test[index] = Guassian_distribution(i)

plt.scatter(xs_cluster_test[:, 0], xs_cluster_test[:, 1])
# plt.show()
assert xs_cluster_test.shape == (100, 2), "You should get 100 2D data points"

# 2.3
def my_plot(xs: np.ndarray):
    """ Plots the K-Means result for different numbers of cluster given 2-dimensional data.
    
    Notes:
        Use the `kmeans_plusplus` function to get initial cluster centers.
    
    Args:
        xs: A 2D numpy array of shape (N, 2) containing N 2-dimensional samples.
    """
    
    plt.figure(figsize=(10, 10))
    n_clusters = [2, 3, 4, 5]  # different numbers of clusters
    
    # iterate over each cluster n in `n_clusters` with index i
    for i, n in enumerate(n_clusters):
        plt.subplot(2, 2, i + 1)
        # YOUR CODE HERE
        centers, indices = kmeans_plusplus(xs, n_clusters = n, random_state = 0)
        centers = my_kmeans(xs, centers, n_iter = 5)
        plt.scatter(xs[:, 0], xs[:, 1], color = 'blue')
        plt.scatter(centers[:, 0], centers[:, 1], color = 'red')
        # raise NotImplementedError()
    # plt.show()

my_plot(xs_cluster_test)

# 3.1
def find_centers(xs: np.ndarray, n_clusters: int=100) -> np.ndarray:
    """ Computes KMeans cluster centers for the given data.
    
    Even though we will use this function to compute centers for our 21-dimensional data points,
    this function should work for arbitrary dimensions.
    
    Notes:
        Use the predefined KMeans algorithm provided by sklearn.
        
    Args:
        xs: A 2D numpy array of shape (N, D) containing N D-dimensional samples.
        n_clusters: Number of clusters to use.
    
    Returns:
        n_clusters D-dimensional cluster centers as a numpy array of shape (n_clusters, D).
    """
    
    # YOUR CODE HERE
    centers, indices = kmeans_plusplus(xs, n_clusters, random_state = 0)
    return centers
    raise NotImplementedError()

# This 100x21 numpy array should contain the cluster centers
xs_centers = find_centers(xs_train_std)
assert xs_centers.shape == (100, 21), "You should get 100 clusters around 21-dimensional centers."

# 3.2
def my_gaussian(r: np.ndarray, sigma: float) -> np.ndarray:
    """ Gaussian probability density function.
    
    The function is applied element-wise.
    
    Args:
        r: A numpy array of arbitrary shape to apply the gaussian to (element-wise).
        sigma: Variance used for normalization.
        
    Returns:
        A numpy array of the same shape as the input r to which the gaussian was applied.
    """
    # YOUR CODE HERE
    phi = np.zeros((np.shape(r)[0], np.shape(r)[1]))

    for i in range(np.shape(phi)[0]):
        for j in range(np.shape(phi)[1]):
            phi[i, j] = pow(np.sqrt(2 * np.pi * pow(sigma, 2)), -1) * np.exp((pow(r[i, j], 2) / (2 * pow(sigma, 2))) * -1)

    return phi
    raise NotImplementedError()

_test_data = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
assert my_gaussian(_test_data, 1).shape == (2, 5), "The Gaussian should not change the shape of the data you apply it to as it must be applied element-wise."

# 3.3
def compute_rbf_features(xs: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """ Computes the feature matrix for the data xs with the given cluster centers and scale.
    
    For the distance, use the euclidean norm.
    Your transformation should not change the order of data points or centers.
    
    Notes:
        You may use np.linalg.norm(x) to get the norm of a vector or matrix x.
        To get the norm along a specified axis a, use np.linalg.norm(x, axis=a)
        
    Args:
        xs: 2D numpy array of shape (N, D) containing N D-dimensional data points.
        centers: 2D numpy array of shape (K, D) containing K D-dimensional centers.
        sigma: Variance used for normalization.
        
    Returns:
        A 2D numpy array of shape (N, K) containing the transformations for each pair of data points and cluster centers.
    """
    # YOUR CODE HERE
    r = np.zeros((np.shape(xs)[0], np.shape(centers)[0]))

    for i in range(np.shape(xs)[0]):
        for j in range(np.shape(centers)[0]):
            temp = np.concatenate((xs[i], centers[j]), axis = 0)
            r[i, j] = np.linalg.norm(temp)

    feature_matrix = my_gaussian(r, sigma)

    # feature_matrix = np.zeros((np.shape(r)[0], np.shape(r)[1]))
    # for i in range(np.shape(feature_matrix)[0]):
    #     for j in range(np.shape(feature_matrix)[1]):
    #         feature_matrix[i, j] = np.exp((pow(r[i, j], 2) / (2 * pow(sigma, 2))) * -1)

    bias_vector = np.ones((np.shape(xs)[0], 1))
    feature_matrix = np.append(bias_vector, feature_matrix, axis = 1)

    return feature_matrix
    raise NotImplementedError()

_test_data = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
_test_centers_1 = np.array([[0, 0, 0, 0, 0.]])
_test_centers_2 = np.array([[0, 0, 0, 0, 0.], [5, 5, 5, 5, 5.]])
_message = "Your transformation should output an array of shape (N, M+1), where N is the number of points and M is the number of cluster centers."
assert compute_rbf_features(_test_data, _test_centers_1, 1).shape == (2, 2), _message
assert compute_rbf_features(_test_data, _test_centers_2, 1).shape == (2, 3), _message

_features_far_data = compute_rbf_features(np.array([[0, 0, 0]]), np.array([[100, 200, 300]]), 1)
_expected_output = np.array([1., 0.])
assert np.allclose(_features_far_data, _expected_output) or np.allclose(_features_far_data[::-1], _expected_output), "For a data point this far from the center, the transformation should be roughly [1., 0.]"

#3.4 
xs_train_gauss = compute_rbf_features(xs_train_std, xs_centers, 25)
_my_weights = my_linear_regression(xs_train_gauss, ys_train_std)

ys_train_pred = ...
xs_valid_gauss = ...
ys_valid_pred = ...
# YOUR CODE HERE
ys_trained_pred = xs_train_gauss @ _my_weights

# xs_valid_centers = find_centers(xs_valid_std)
# xs_valid_gauss = compute_rbf_features(xs_valid_std, xs_valid_centers, 25)
xs_valid_gauss = compute_rbf_features(xs_valid_std, xs_centers, 25)
# _my_weights = my_linear_regression(xs_valid_gauss, ys_valid_std)
ys_valid_pred = xs_valid_gauss @ _my_weights
# raise NotImplementedError()

_mse = my_mse(ys_valid_std, ys_valid_pred)

print(f"Your validation MSE should be roughly 18.3 and it is {_mse}")
