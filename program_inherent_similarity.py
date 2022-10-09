from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import sqrt
from sklearn import metrics
from torch import nn

spec17_clusters = [
    [508, 544],
    [519, 549],
    [507],
]


def get_distribution_distance(U, V):
    # P = np.array([3,5,2,1,3])
    # Q = np.array([2,3,4,5,0])
    # dists = [i for i in range(len(A))]
    from scipy.stats import wasserstein_distance
    U = [float(u) for u in U]
    V = [float(u) for u in V]
    D = wasserstein_distance(U, V)
    #print(D)
    return D


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


def get_domain_distance(domain_feature_0, domain_feature_1, domain_loss='Sinkhorn'):
    domain_feature_0 = torch.Tensor(domain_feature_0)
    domain_feature_1 = torch.Tensor(domain_feature_1)

    if 'mkmmd' == domain_loss:
        from MultipleKernelMaximumMeanDiscrepancy import MultipleKernelMaximumMeanDiscrepancy
        transfer_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=False
        )
    elif 'mkmmd_2' == domain_loss:
        from MMDLoss import MMDLoss
        transfer_loss = MMDLoss(kernel_type='rbf', kernel_mul=2.0, kernel_num=3)  # 'linear'
    elif 'Sinkhorn' == domain_loss:
        transfer_loss = SinkhornDistance(eps=1e-4, max_iter=10, )
    else:
        transfer_loss = None

    if 'mkmmd' == domain_loss or 'mkmmd_2' == domain_loss:
        transfer_loss1 = transfer_loss(domain_feature_0, domain_feature_1)
        if torch.isnan(transfer_loss1):
            print(f"error: train transfer_loss ({domain_loss}) is NaN!!!!")
            transfer_loss1 = 0.1
        transfer_loss = transfer_loss1 * transfer_loss1
        domain_loss_lamda_iter = 1  # min(2000, int(loss_label/cost)) # trade-off
    elif 'Sinkhorn' == domain_loss:
        # src_domain_feature = torch.reshape(torch.Tensor(src_domain_feature), (src_domain_feature.size(0), src_domain_feature.size(1)))
        # dst_domain_feature = torch.reshape(torch.Tensor(dst_domain_feature), (len(dst_domain_feature), len(dst_domain_feature[0])))
        transfer_loss, pi, C = transfer_loss(domain_feature_0, domain_feature_1)
        domain_loss_lamda_iter = 5000  # min(2000, int(loss_label/cost)) # trade-off
    elif 'wasserstein' == domain_loss:
        # wasserstein is same with Sinkhorn,
        transfer_loss = get_distribution_distance(domain_feature_0, domain_feature_1)
    else:
        transfer_loss = 0
        domain_loss_lamda_iter = 0

    return transfer_loss


'''
if 0:
    #KL is only a scalar, not for gradiant
    from metric_function import asymmetricKL
    CPI_histogram_1 = CPI_histogram["500.1-refrate-1"]
    CPI_histogram_2 = CPI_histogram["519.1-refrate-1"]
    print(f"CPI_histogram_1={CPI_histogram_1}")
    print(f"CPI_histogram_2={CPI_histogram_2}")
    kl = asymmetricKL(CPI_histogram_1,CPI_histogram_2)
    print(kl)
'''


def get_all_distance(metric_iter='CPI', domain_loss='Sinkhorn'):       # , 'mkmmd_2']:
    from plot import case_names
    metric_iter_map_id = 0 if 'CPI' == metric_iter else 1
    distance_data = np.zeros((len(case_names), len(case_names)))
    data_config_str = metric_iter + '_' + domain_loss
    filename = 'domain_distance_data/distance_' + data_config_str + '.txt'
    print(f"get_all_distance: filename={filename}")
    distance_file = open(filename, 'w')

    for case_name_row_iter_id, case_name_row_iter in enumerate(case_names):
        from program_inherent_similarity import get_domain_distance
        from simulation_metrics import get_dataset
        v, domain_feature_0 = get_dataset(case_name=case_name_row_iter)
        domain_feature_0 = np.asarray(domain_feature_0)[:, metric_iter_map_id]
        domain_feature_0 = np.reshape(domain_feature_0, (-1, 1))

        for case_name_col_iter_id, case_name_col_iter in enumerate(case_names[:case_name_row_iter_id+1]):
            v, domain_feature_1 = get_dataset(case_name=case_name_col_iter)
            domain_feature_1 = np.asarray(domain_feature_1)[:, metric_iter_map_id]
            domain_feature_1 = np.reshape(domain_feature_1, (-1, 1))

            distance = get_domain_distance(domain_feature_0, domain_feature_1, domain_loss=domain_loss).item()
            print(f"[{case_name_row_iter_id}][{case_name_col_iter_id}] {case_name_col_iter} distance= {distance}")
            distance_file.write(f"{distance} ")
            distance_data[case_name_row_iter_id, case_name_col_iter_id] = distance
        distance_file.write('\n')

    distance_max = np.max(distance_data)
    for case_name_row_iter_id in range(len(case_names)):
        for case_name_col_iter_id in range(case_name_row_iter_id+1, len(case_names)):
            distance_data[case_name_row_iter_id, case_name_col_iter_id] = distance_max

    distance_file.close()
    return distance_data, data_config_str


def get_all_distance_from_file(metric_iter='CPI', domain_loss='Sinkhorn'):
    from plot import case_names
    distance_data = np.zeros((len(case_names), len(case_names)))
    data_config_str = metric_iter + '_' + domain_loss
    filename = 'domain_distance_data/distance_' + data_config_str + '.txt'
    print(f"get_all_distance: filename={filename}")
    distance_file = open(filename, 'r')

    row_index = 0
    for each_line in distance_file:
        #a = each_line.split(' ')[:-2]
        row_value = each_line.split(' ')[:-1]
        if len(row_value):
            row_value = [float(i) for i in row_value]
            row_value[-1] = 1e-10
            distance_data[:len(row_value), row_index] = row_value
            distance_data[row_index, :len(row_value)] = row_value
            row_index += 1
    distance_file.close()

    '''
    distance_max = np.max(distance_data)
    for case_name_row_iter_id in range(len(case_names)):
        for case_name_col_iter_id in range(case_name_row_iter_id+1, len(case_names)):
            distance_data[case_name_row_iter_id, case_name_col_iter_id] = distance_max
    '''
    return distance_data, data_config_str


def get_domain_cluster(distance_data):
    cluster_labels = get_domain_cluster_hierarchical(distance_data)
    print(f"cluster_labels={cluster_labels}")
    return cluster_labels


def get_domain_cluster_keamns(distance_data):
    from sklearn.cluster import KMeans
    scores = []
    labels_list = []
    for n_clusters in range(3, 10):
        kmeans = KMeans(n_clusters=4)
        cluster_labels = kmeans.fit_predict(X=distance_data)
        print(f"get_domain_cluster labels= {cluster_labels}")
        score = metrics.calinski_harabasz_score(distance_data, cluster_labels)
        labels_list.append(cluster_labels)
        #print(kmeans.cluster_centers_)
        #print(kmeans.inertia_)
        print(f"silhouette_score={metrics.silhouette_score(distance_data, cluster_labels)}")
        scores.append(score)
    bset_index = np.argmax(scores)
    best_n_clusters = bset_index + 3
    print(f"best_n_clusters={best_n_clusters}")
    best_cluster_labels = labels_list[bset_index]
    return best_cluster_labels


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram
    dendrogram(linkage_matrix, **kwargs)


def get_domain_cluster_hierarchical_all(distance_data, data_config_str=None):
    from sklearn.cluster import AgglomerativeClustering
    scores = []
    labels_list = []
    model_list = []
    for n_clusters in range(1):
        cluster_algo = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
        cluster_labels = cluster_algo.fit_predict(X=distance_data)
        labels_list.append(cluster_labels)
        model_list.append(cluster_algo)
    print(f"scores={scores}")
    best_index = 0 #np.argmax(scores)
    #best_n_clusters = best_index + 4
    #print(f"best_n_clusters={best_n_clusters}")
    best_cluster_labels = labels_list[best_index]
    model = model_list[best_index]
    plt.figure()
    plot_dendrogram(model, truncate_mode='level', p=7) #p is level
    plt.savefig('fig/cluster_' + data_config_str + '.png')
    return best_cluster_labels


def get_domain_cluster_hierarchical(distance_data):
    from sklearn.cluster import AgglomerativeClustering
    n_clusters = 3
    print(f"n_clusters={n_clusters}")
    cluster_algo = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete') # average, ['ward', 'complete', 'average', 'single']
    cluster_labels = cluster_algo.fit_predict(X=distance_data)
    return cluster_labels


if __name__ == '__main__':
    metric_name = "CPI"
    #metric_name = "Power"
    #domain_loss = 'Sinkhorn'
    #domain_loss = 'mkmmd_2'
    domain_loss = 'wasserstein'
    #distance_data, data_config_str = get_all_distance(metric_iter=metric_name, domain_loss=domain_loss)
