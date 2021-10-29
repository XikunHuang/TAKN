import networkit
from networkit import *
import matplotlib.pyplot as plt
import numpy as np
import network_scanner.networkit_based.networkit_util as networkit_util


def plot_degree_dist(net, label, outpath):
    """
    Plot degree distribution.
    Args:
        net: networkit graph object
        label: network name
        outpath:

    Returns: unique degree list

    """
    unique_deg, unique_cnt = networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='all')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(unique_deg, unique_cnt, 'b*', label=label)
    # ax.set_title('Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-degree-distribution.eps')
    return unique_deg


def plot_indeg_dist(net, label, outpath):
    """
    Plot in-degree distribution
    """
    unique_deg, unique_cnt = networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='in')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(unique_deg, unique_cnt, 'g*', label=label)
    # ax.set_title('In-Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-indegree-distribution.eps')
    return unique_deg


def plot_outdeg_dist(net, label, outpath):
    """
    Plot out-degree distribution
    """
    unique_deg, unique_cnt = networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='out')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(unique_deg, unique_cnt, 'r*', label=label)
    # ax.set_title('Out-Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-outdegree-distribution.eps')
    return unique_deg


def plot_ccum_degree_dist(net, label, outpath, degree_type='all'):
    """
    Plot complementary cumulative degree distribution
    """
    unique_deg, unique_cnt = networkit_util.get_cc_deg_dist(net, degree_type)
    title = {'all': '', 'in': 'In', 'out': 'Out'}
    outfile_name = {'all': 'cc', 'in': 'cc-in', 'out': 'cc-out'}
    marker_color = {'all': 'b', 'in': 'g', 'out': 'r'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(unique_deg, unique_cnt, color=marker_color[degree_type], marker='*', label=label)
    # ax.set_title('Complementary Cumulative ' + title[degree_type] + '-Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x>=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-' + outfile_name[degree_type] + '-degree-distribution.eps')
    return ax


def plot_out_in_degree_comparision(net, label, outpath):
    """
    Plot outdegree-indegree comparision
    """
    zipped_seq = [(net.degreeOut(node), net.degreeIn(node)) for node in net.iterNodes()]
    out_deg_seq, in_deg_seq = zip(*zipped_seq)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(out_deg_seq, in_deg_seq, 'b.', markersize=1)
    # ax.set_title('Outdegree/Indegree comparision')
    ax.set_xlabel('Out-degree')
    ax.set_ylabel('In-degree')
    plt.savefig(outpath + label + '-outdegree-indegree-comparision.eps')


def plot_cum_clustering_dist(net, label, outpath, turbo):
    """
    Plot cumulative distribution of clustering coefficient of nodes. ONLY support Undirected graph.
    Args:
        net:
        label:
        outpath:
        turbo: There are two algorithms available. The trivial (parallel) algorithm needs only a small amount of additional memory.
                The turbo mode adds a (sequential, but fast) pre-processing step using ideas from [0].
                This reduces the running time significantly for most graphs. However, the turbo mode needs O(m) additional memory.
                 In practice this should be a bit less than half of the memory that is needed for the graph itself.
                 The turbo mode is particularly effective for graphs with nodes of very high degree and a very skewed degree distribution.

                    [0] Triangle Listing Algorithms: Back from the Diversion Mark Ortmann and Ulrik Brandes 2014 Proceedings of the Sixteenth Workshop on Algorithm Engineering and Experiments (ALENEX). 2014, 1-8


    Returns:

    """
    net.removeSelfLoops()
    local_cc = networkit.centrality.LocalClusteringCoefficient(net, turbo)
    local_cc.run()
    unique_cc, unique_cc_cnt = np.unique(local_cc.scores(), return_counts=True)
    unique_cc_cumcnt = np.cumsum(unique_cc_cnt)/sum(unique_cc_cnt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis([0, 1, 0, 1])
    ax.plot(unique_cc, unique_cc_cumcnt, 'b-')
    # ax.set_title('Cumulative distribution of clustering coefficient of nodes')
    ax.set_xlabel('Local clustering coefficient c')
    ax.set_ylabel('p(x <= c)')
    plt.savefig(outpath + label + "-cc-distribution.eps")


def plot_assorsativity(net, label, outpath, degree_type='all'):
    """
    Plot degree assorsativity. ONLY support Undirected graph.
    """
    deg_seq, nbdeg_seq = networkit_util.get_deg_nbdeg(net, degree_type)
    title = {'all': '', 'in': 'In-', 'out': 'Out-'}
    outfile_name = {'all': 'assorsativity', 'in': 'assorsativity-in', 'out': 'assorsativity-out'}
    marker_color = {'all': 'b', 'in': 'g', 'out': 'r'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(deg_seq, nbdeg_seq, marker_color[degree_type]+'.', markersize=1)
    # ax.set_title(title[degree_type] + 'Degreee assortativity')
    ax.set_xlabel(degree_type + '-degree')
    ax.set_ylabel('average neighbour ' + degree_type + '-degree')
    plt.savefig(outpath + label + '-' + outfile_name[degree_type] + '-assortativity-plot.eps')


def plot_connected_component_dist(net, label, outpath):
    """
    Plot connected components of undirected graph
    """
    cc = networkit_util.get_connected_components(net)
    cc_size, cc_cnt = np.unique(cc, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(cc_size, cc_cnt, 'r*', label=label)
    # ax.set_title('Connected components distribution')
    ax.set_xlabel('size')
    ax.set_ylabel('count')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-connected-component-distribution.eps')
    return cc


def plot_wcc_dist(net, label, outpath):
    """
    Plot weakly connected components of directed graph
        
    Return:
        weakly connected component size distribution
    """
    wcc = networkit_util.get_wcc(net)
    wcc_size, wcc_cnt = np.unique(wcc, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(wcc_size, wcc_cnt, 'b*', label=label)
    # ax.set_title('Weakly connected components distribution')
    ax.set_xlabel('size')
    ax.set_ylabel('count')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-wcc-distribution.eps')
    return wcc


def plot_scc_dist(net, label, outpath):
    """
    Plot strongly connected components of directed graph

    Return:
        strongly connected component size distribution

    """
    scc = networkit_util.get_scc(net)
    scc_size, scc_cnt = np.unique(scc, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(scc_size, scc_cnt, 'g*', label=label)
    # ax.set_title('Strongly connected components distribution')
    ax.set_xlabel('size')
    ax.set_ylabel('count')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-scc-distribution.eps')
    return scc


def plot_betweeness(net, label, outpath):
    """
    Plot cumulative betweenss centrality
    """
    _, betweeness_values = networkit_util.get_betweeness(net, label, outpath)
    unique_value, unique_cnt = np.unique(betweeness_values, return_counts=True)
    unique_cumcnt = np.cumsum(unique_cnt) / sum(unique_cnt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(unique_value, unique_cumcnt, 'b.')
    # ax.set_title('Cumulative distribution of betweeness centrality of nodes')
    ax.set_xlabel('betweeness centrality b')
    ax.set_ylabel('p(x <= b)')
    plt.savefig(outpath + label + "-betweeness-distribution.eps")


def plot_eigenvector_centrality(net, label, outpath):
    """
    Plot cumulative eigenvector centrality
    """
    _, eigenvector_centrality_values = networkit_util.get_eigenvector_centrality(net, label, outpath)
    unique_value, unique_cnt = np.unique(eigenvector_centrality_values, return_counts=True)
    unique_cumcnt = np.cumsum(unique_cnt) / sum(unique_cnt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(unique_value, unique_cumcnt, 'g.')
    # ax.set_title('Cumulative distribution of eigenvector centrality of nodes')
    ax.set_xlabel('eigenvector centrality e')
    ax.set_ylabel('p(x <= e)')
    plt.savefig(outpath + label + "-eigenvector-distribution.eps")


def plot_pagerank(net, label, outpath):
    """
    Plot cumulative pagerank centrality
    """
    _, pagerank_values = networkit_util.get_pagerank(net, label, outpath)
    unique_value, unique_cnt = np.unique(pagerank_values, return_counts=True)
    unique_cumcnt = np.cumsum(unique_cnt) / sum(unique_cnt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(unique_value, unique_cumcnt, 'r.')
    # ax.set_title('Cumulative distribution of pagerank of nodes')
    ax.set_xlabel('pagerank value v')
    ax.set_ylabel('p(x <= v)')
    plt.savefig(outpath + label + "-pagerank-distribution.eps")


def plot_ccum_centrality_dist(centrality_filename, label, outpath, centrality_name):
    """
    Plot complementary cumulative centrality distribution
    """
    unique_val, unique_cc_prob = networkit_util.get_cc_centrality_distr(centrality_filename)
    centrality_style = {'eigenvector-centrality': 'c*', 'pagerank': 'g*', 'hub': 'r*', 'authority': 'm*', 'betweeness': 'b*'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.loglog(unique_val, unique_cc_prob, centrality_style[centrality_name], label=label + '-' + centrality_name)
    ax.set_xlabel('v')
    ax.set_ylabel('P(x>=v)')
    plt.savefig(outpath + label + '-' + centrality_name + '-distribution.eps')
    return ax


def plot_hop_dist(net, label, outpath):
    """
    Plot hop distribution.only support connected graph
    """
    dist, proportion = networkit_util.get_hop_distr(net, label, outpath)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dist, proportion, 'g*', label=label)
    ax.set_xlabel('distance d')
    ax.set_ylabel('p(x<=d)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-hop.eps')


def plot_degree_dist_fromfile(filepath, label, degree_type, outpath):
    """
    Plot degree distribution
    """
    degree_seq = []
    degree_content = np.genfromtxt(filepath, delimiter='\t')
    for item in degree_content:
        degree_seq.append(item[1])
    unique_deg, unique_cnt = np.unique(degree_seq, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if degree_type == 'reciprocity':
        ax.loglog(unique_deg, unique_cnt, 'b*', label=label)
    else:
        ax.loglog(unique_deg, unique_cnt, 'b*', label=label)
    # ax.set_title('Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-' + degree_type + 'degree-distribution.eps')
    return unique_deg


def plot_ccum_degree_dist_fromfile(filepath, label, degree_type, outpath):
    """
    Plot complementary cumulative degree distribution
    Args:
        filepath:
        label:
        degree_type: '', 'in', 'out', 'reciprocity'
        outpath:

    Returns:

    """
    degree_seq = []
    degree_content = np.genfromtxt(filepath, delimiter='\t')
    for item in degree_content:
        degree_seq.append(item[1])
    unique_deg, unique_cnt = np.unique(degree_seq, return_counts=True)
    tmp_sum = sum(unique_cnt)
    unique_cc_cnt = [sum(unique_cnt[i:]) / tmp_sum for i in range(len(unique_cnt))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if degree_type == 'reciprocity':
        ax.loglog(unique_deg, unique_cc_cnt, 'b*', label=label)
    else:
        ax.loglog(unique_deg, unique_cc_cnt, 'b*', label=label)
    # ax.set_title('Degree distribution')
    ax.set_xlabel('k')
    ax.set_ylabel('P(x>=k)')
    # ax.legend(loc='best')
    plt.savefig(outpath + label + '-' + degree_type + 'degree-complementary-cumulative-distribution.eps')
    return unique_deg

