import networkit as nk
import numpy as np
import scipy.sparse.csgraph
import scipy.sparse.linalg
import scipy.stats
import random
import powerlaw
import matplotlib.pyplot as plt
import os
import logging


def read_edgelist(filepath, separator, firstNode, continuous, directed):
    # Read network from edge list.
    # ref:https://github.com/networkit/networkit/issues/725
    if not continuous:
        firstNode = 0

    net_reader = nk.graphio.EdgeListReader(separator=separator,
                                           firstNode=firstNode,
                                           continuous=continuous,
                                           directed=directed)
    net = net_reader.read(filepath)
    # print(net.numberOfNodes(), net.numberOfEdges())
    # for item in net.iterNodes():
    #     print(item)
    # for e in net.iterEdges():
    #     print(e)
    return net


# def readnetwork(filepath, directed):
#     """
#     read network from edgelist file
#
#     :param filepath: edgelist file
#
#     :param directed: boolean, treat the network as directed or not
#
#     :return:
#
#     """
#     net_reader = nk.graphio.EdgeListReader(separator='\t', firstNode=1, continuous=False, directed=directed)
#     net = net_reader.read(filepath)
#     return net


def get_deg_dist(net, degree_type='all'):
    """
    Get (in/out)degree distribution.
    Args:
        net:
        degree_type: [all, in, out]
    Returns:

    """
    if degree_type not in ['all', 'in', 'out']:
        print("ERROR degree type.")
        return
    if degree_type == 'all':
        ret = [net.degree(node) for node in net.iterNodes()]
        return np.unique(ret, return_counts=True)
    if degree_type == 'in':
        ret = [net.degreeIn(node) for node in net.iterNodes()]
        return np.unique(ret, return_counts=True)
    if degree_type == 'out':
        ret = [net.degreeOut(node) for node in net.iterNodes()]
        return np.unique(ret, return_counts=True)


def get_and_write_deg_dist(net, label, outpath, degree_type='all'):
    """
    Get (in/out)degree distribution
    """
    if degree_type not in ['all', 'in', 'out']:
        print("ERROR degree type.")
        return
    if degree_type == 'all':
        ret = []
        degree_file = open(outpath + label + '-falseid-degree', 'w')
        for node in net.iterNodes():
            deg = net.degree(node)
            degree_file.write(str(node) + '\t' + str(deg) + '\n')
            ret.append(deg)
        degree_file.close()
        return np.unique(ret, return_counts=True)
    if degree_type == 'in':
        ret = []
        indegree_file = open(outpath + label + '-falseid-indegree', 'w')
        for node in net.iterNodes():
            indeg = net.degreeIn(node)
            indegree_file.write(str(node) + '\t' + str(indeg) + '\n')
            ret.append(indeg)
        # ret = [net.degreeIn(node) for node in net.iterNodes()]
        indegree_file.close()
        return np.unique(ret, return_counts=True)
    if degree_type == 'out':
        ret = []
        outdegree_file = open(outpath + label + '-falseid-outdegree', 'w')
        for node in net.iterNodes():
            outdeg = net.degreeOut(node)
            outdegree_file.write(str(node) + '\t' + str(outdeg) + '\n')
            ret.append(outdeg)
        # ret = [net.degreeOut(node) for node in net.iterNodes()]
        outdegree_file.close()
        return np.unique(ret, return_counts=True)


def get_cc_deg_dist(net, degree_type='all'):
    """
    Get complementary cumulative degree distribution
   """
    uniqe_ele, unique_cnt = get_deg_dist(net, degree_type)
    tmp_sum = sum(unique_cnt)
    unique_cc_cnt = [sum(unique_cnt[i:])/tmp_sum for i in range(len(unique_cnt))]
    # Another solution
    # ant = tmp_sum - np.cumsum(unique_cnt) + unique_cnt
    # ant = ant/tmp_sum
    return uniqe_ele, unique_cc_cnt


def get_deg_nbdeg(net, degree_type='all'):
    """
    Get (in/out)degree vs neighbor (in/out)degree
    """
    deg_seq = []
    nbdeg_seq = []
    if degree_type == 'all':
        for node in net.iterNodes():
            node_deg = net.degree(node)
            deg_seq.append(node_deg)
            avg_deg = 0
            node_neighbors = net.iterNeighbors(node)
            if node_deg != 0:
                for ele in node_neighbors:
                    # print(net.degree(ele))
                    avg_deg = avg_deg + net.degree(ele)
                # print(avg_deg, len(node_neighbors))
                avg_deg = avg_deg / node_deg
            nbdeg_seq.append(avg_deg)
        return deg_seq, nbdeg_seq


def get_reciprocity(net):
    num_edges = net.numberOfEdges()
    colinks = 0
    for ele in net.iterEdges():
        if net.hasEdge(ele[1], ele[0]):
            colinks += 1
    return colinks / num_edges


def _node_degree_xy(net, x='out', y='in'):
    for u in net.iterNodes():
        degu = net.degreeOut(u)
        neighbors = (nbr for nbr in net.iterNeighbors(u))
        for v in neighbors:
            degv = net.degreeIn(v)
            yield degu, degv


def _node_degree_neighbourdegree(net):
    for u in net.iterNodes():
        deg_u = net.degree(u)
        neighbors_u = net.neighbors(u)
        neighbors_num = len(neighbors_u)
        if not neighbors_num:
            continue
        average_degree = 0
        for nb in neighbors_u:
            average_degree += net.degree(nb)
        average_degree = average_degree / neighbors_num
        yield deg_u, average_degree


def get_assorsativity(net):
    xy = _node_degree_xy(net)
    x, y = zip(*xy)
    return scipy.stats.pearsonr(x, y)[0]


def get_connected_components(net):
    """
    Get sizes of connected components
    """
    cc = nk.components.ConnectedComponents(net)
    cc.run()
    num_of_cc = cc.numberOfComponents()
    sizes_of_cc = cc.getComponentSizes()
    cc = [sizes_of_cc[k] for k in sizes_of_cc]
    return cc


def get_wcc(net):
    """
    Get sizes of weakly connected components
    """
    wcc = nk.components.WeaklyConnectedComponents(net)
    wcc.run()
    num_of_wcc = wcc.numberOfComponents()
    sizes_of_wcc = wcc.getComponentSizes()
    wcc = [sizes_of_wcc[k] for k in sizes_of_wcc]
    return wcc


def get_scc(net):
    """
    Get sizes of strongly connected components
    """
    scc = nk.components.StronglyConnectedComponents(net)
    scc.run()
    num_of_scc = scc.numberOfComponents()
    scc_partitions = scc.getPartition()
    # reference https://networkit.iti.kit.edu/api/networkit.html?highlight=partition#networkit.Partition
    scc = scc_partitions.subsetSizes()
    return scc


def get_betweeness(net, label, outpath):
    betweeness = nk.centrality.ApproxBetweenness(net, epsilon=0.01, delta=0.1, universalConstant=1.0)
    betweeness.run()
    betweeness_file = outpath + label + '-betweeness-falseid-value'
    betweeness_ranking = open(betweeness_file, 'w')
    nodes_id, betweeness_values = zip(*betweeness.ranking())
    for i in range(len(nodes_id)):
        betweeness_ranking.write(str(nodes_id[i]) + '\t' + str(betweeness_values[i]) + '\n')
    betweeness_ranking.close()
    return nodes_id, betweeness_values


def get_eigenvector_centrality(net, label, outpath):
    eigenvector_centrality = nk.centrality.EigenvectorCentrality(net,  tol=1e-9)
    eigenvector_centrality.run()
    eigenvector_file = outpath + label + '-eigenvector-centrality-falseid-value'
    nodes_id, eigenvector_centrality_values = zip(*eigenvector_centrality.ranking())
    eigenvector_ranking = open(eigenvector_file, 'w')
    for i in range(len(nodes_id)):
        eigenvector_ranking.write(str(nodes_id[i]) + '\t' + str(eigenvector_centrality_values[i]) + '\n')
    eigenvector_ranking.close()
    return nodes_id, eigenvector_centrality_values


def get_pagerank(net, label, outpath):
    pagerank_value = nk.centrality.PageRank(net, damp=0.85, tol=1e-9)
    pagerank_value.run()
    pagerank_file = outpath + label + '-pagerank-falseid-value'
    nodes_id, pagerank_values = zip(*pagerank_value.ranking())
    pagerank_ranking = open(pagerank_file, 'w')
    for i in range(len(nodes_id)):
        pagerank_ranking.write(str(nodes_id[i]) + '\t' + str(pagerank_values[i]) + '\n')
    pagerank_ranking.close()
    return nodes_id, pagerank_values


def get_cc_centrality_distr(centrality_filename):
    """
    Get complementary cumulative centrality distribution
    """
    centrality_file = open(centrality_filename, 'r')
    val = []
    while True:
        line = centrality_file.readline()
        if not line:
            break
        splited_line = line.strip().split('\t')
        val.append(float(splited_line[1]))
    centrality_file.close()
    unique_val, unique_cnt = np.unique(val, return_counts=True)
    tmp_sum = sum(unique_cnt)
    # unique_cc_cnt = [sum(unique_cnt[i:]) / tmp_sum for i in range(len(unique_cnt))]
    unique_cc_cnt = tmp_sum - np.cumsum(unique_cnt) + unique_cnt
    unique_cc_prob = unique_cc_cnt / tmp_sum
    return unique_val, unique_cc_prob


def get_hop_distr(net, label, outpath):
    hop_appro = nk.distance.HopPlotApproximation(net)
    hop_appro.run()
    hop_dist = hop_appro.getHopPlot()
    dist = []
    proportion = []
    for d in sorted(hop_dist):
        dist.append(d)
        proportion.append(hop_dist[d])
    hop_file = open(outpath + label + "-hop-distribution", 'w')
    for i in range(len(dist)):
        hop_file.write(str(dist[i]) + '\t' + str(proportion[i]) + '\n')
    hop_file.close()
    return dist, proportion


def get_lcc_subgraph(net):
    """
    Get the largest connected component
    """
    cc = nk.components.ConnectedComponents(net)
    cc.run()
    partition = cc.getPartition()
    scc_size_map = partition.subsetSizeMap()
    lcc_size = 0
    lcc_id = 0
    for id in scc_size_map.keys():
        if scc_size_map[id] > lcc_size:
            lcc_size = scc_size_map[id]
            lcc_id = id
    print(lcc_id, lcc_size)
    lcc_index = list(partition.getMembers(lcc_id))
    result_subgraph = nk.graphtools.subgraphFromNodes(net, lcc_index)
    # print(result_sbgraph.numberOfNodes())
    # print(result_sbgraph.numberOfEdges())
    return result_subgraph


def get_lscc_subgraph(net):
    """
    Get the largest strongly connected component
    Return:
         subgraph induced by largest strongly connected component
    """
    scc = nk.components.StronglyConnectedComponents(net)
    scc.run()
    scc_partition = scc.getPartition()
    scc_size_map = scc_partition.subsetSizeMap()
    lscc_size = 0
    lscc_id = 0
    for id in scc_size_map.keys():
        if scc_size_map[id] > lscc_size:
            lscc_size = scc_size_map[id]
            lscc_id = id
    print(lscc_id, lscc_size)
    lscc_index = list(scc_partition.getMembers(lscc_id))
    print(len(lscc_index))
    result_subgraph = nk.graphtools.subgraphFromNodes(net, lscc_index)
    print(result_subgraph.numberOfNodes())
    print(result_subgraph.numberOfEdges())
    return result_subgraph


def write_map_node_id(net_reader, label, outpath):
    """
    Write mapNodeId to file.
    
    Return:
         dict. key is real id in origin id pair file, and value is id used in program
    """
    result_path = outpath + label + '-map-node-ids'
    out = open(result_path, 'w')
    map_node_ids = nk.graphio.EdgeListReader.getNodeMap(net_reader)
    for raw_id in map_node_ids:
        processed_id = map_node_ids[raw_id]
        out.write(str(raw_id) + '\t' + str(processed_id) + '\n')
    out.close()
    return result_path


def get_falseid_trueid_map(map_filename):
    """
    Get node id map
    
    Return:
         key is id used in program, and value is real id in origin id pair file
    """
    map_file = open(map_filename, 'r')
    result = dict()
    while True:
        line = map_file.readline()
        if not line:
            break
        split_line = line.strip().split(sep='\t')
        result[split_line[1]] = split_line[0]
    map_file.close()
    return result


def get_average_shortest_path_appro(connected_net, sample_num):
    """
    Estimate average shortest path.
    BUG: subgraph stores unnecessary information of origin graph. BFS can detect it.
    """
    sample = []
    nodes = connected_net.iterNodes()
    for i in range(sample_num):
        sample.append(nodes[random.randint(1, len(nodes))])
    sampled_path_length = 0

    for source in sample:
        bfs = nk.distance.BFS(connected_net, source=source, storePaths=False)
        bfs.run()
        for target in nodes:
            sampled_path_length = sampled_path_length + bfs.distance(target)
    result = sampled_path_length / (len(sample) * len(nodes))
    return result


def falseid_to_trueid(false_id_list, map_filename):
    """
    False id to real id.
    Args:
        false_id_list:
        map_filename: map filename. In this map file, the first column is true id and the second column is false id.

    Returns:
        true id list
    """
    map_file = open(map_filename, 'r')
    id_name_map = dict()
    while True:
        line = map_file.readline()
        if not line:
            break
        splited_line = line.strip().split('\t')
        id_name_map[splited_line[1]] = splited_line[0]
    map_file.close()
    trueid_list = []
    for id in false_id_list:
        trueid = id_name_map.get(id)
        trueid_list.append(trueid)
    return trueid_list


def trueid_to_name(id_list, index_filename):
    """
    Id to name.
    """
    index_file = open(index_filename, 'r')
    id_name_map = dict()
    while True:
        line = index_file.readline()
        if not line:
            break
        splited_line = line.strip().split('\t')
        id_name_map[splited_line[0]] = splited_line[1]
    index_file.close()
    name_list = []
    for id in id_list:
        name = id_name_map.get(id)
        name_list.append(name)
    return name_list


def get_reciprocity_deg(net):
    """
    Get reciprocity degree of nodes.
    
    Args:
        net ([type]): directed graph
    
    Returns:
        [type]: list. [[node_falseid, reciprocity_deg], [1,32],...]
    """
    is_directed = net.isDirected()
    if not is_directed:
        print("error. only directed graph allowed.")
        return
    reciprocity_degree_list = []
    for node in net.iterNodes():
        cur_out_neighbors = net.neighbors(node)
        cur_reciprocity_degree = 0
        if len(cur_out_neighbors):
            for item in cur_out_neighbors:
                if node in net.neighbors(item):
                    cur_reciprocity_degree = cur_reciprocity_degree + 1
        reciprocity_degree_list.append([node, cur_reciprocity_degree])
    return reciprocity_degree_list


def get_and_write_reciprocity_degree(net, map_filename, out_filename):
    """
    Get and write reciprocity degree. [node_trueid  reciprocity_degree]
    
    Args:
        net ([type]): [description]
        map_filename ([type]): map file. The 1st col is true id. The 2nd col is false id.
        out_filename ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    falseid_trueid_map = get_falseid_trueid_map(map_filename)
    is_directed = net.isDirected()
    if not is_directed:
        print("error. only directed graph allowed.")
        return
    reciprocity_degree_list = []
    for node in net.iterNodes():
        cur_out_neighbors = net.neighbors(node)
        cur_reciprocity_degree = 0
        if len(cur_out_neighbors):
            for item in cur_out_neighbors:
                if node in net.neighbors(item):
                    cur_reciprocity_degree = cur_reciprocity_degree + 1
        reciprocity_degree_list.append([falseid_trueid_map[str(node)], cur_reciprocity_degree])
    with open(out_filename, 'w') as out_file:
        for item in reciprocity_degree_list:
            out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')


def write_trueid_reciprocity_degree(net_reader, net, label, outpath):
    """
    Get [trueid reciprocitydegree]
    """
    map_filename = write_map_node_id(net_reader, label, outpath + 'reciprocity-degree-')
    falseid_trueid_map = get_falseid_trueid_map(map_filename)
    out_filename = outpath + label + '-trueid-reciprocitydegree'
    is_directed = net.isDirected()
    if not is_directed:
        print("Error! Only directed graph allowed.")
        return
    reciprocity_degree_list = []
    for node in net.iterNodes():
        cur_out_neighbors = net.neighbors(node)
        cur_reciprocity_degree = 0
        if len(cur_out_neighbors):
            for item in cur_out_neighbors:
                if node in net.neighbors(item):
                    cur_reciprocity_degree = cur_reciprocity_degree + 1
        reciprocity_degree_list.append([falseid_trueid_map[str(node)], cur_reciprocity_degree])
    with open(out_filename, 'w') as out_file:
        for item in reciprocity_degree_list:
            out_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')


def write_trueid_deg(net_reader, net, label, outpath, degree_type):
    """
    Get [trueid degree]
    """
    if degree_type not in ['all', 'in', 'out']:
        print("ERROR degree type.")
        return
    map_filename = write_map_node_id(net_reader, label, outpath + degree_type + 'degree-')
    falseid_trueid_map = get_falseid_trueid_map(map_filename)
    if degree_type == 'all':
        ret = []
        degree_file = open(outpath + label + '-trueid-degree', 'w')
        for node in net.iterNodes():
            deg = net.degree(node)
            degree_file.write(falseid_trueid_map[str(node)] + '\t' + str(deg) + '\n')
            ret.append(deg)
        degree_file.close()
        return np.unique(ret, return_counts=True)
    if degree_type == 'in':
        ret = []
        indegree_file = open(outpath + label + '-trueid-indegree', 'w')
        for node in net.iterNodes():
            indeg = net.degreeIn(node)
            indegree_file.write(falseid_trueid_map[str(node)] + '\t' + str(indeg) + '\n')
            ret.append(indeg)
        # ret = [net.degreeIn(node) for node in net.iterNodes()]
        indegree_file.close()
        return np.unique(ret, return_counts=True)
    if degree_type == 'out':
        ret = []
        outdegree_file = open(outpath + label + '-trueid-outdegree', 'w')
        for node in net.iterNodes():
            outdeg = net.degreeOut(node)
            outdegree_file.write(falseid_trueid_map[str(node)] + '\t' + str(outdeg) + '\n')
            ret.append(outdeg)
        # ret = [net.degreeOut(node) for node in net.iterNodes()]
        outdegree_file.close()
        return np.unique(ret, return_counts=True)


def power_law_analysis(filename, label, outpath, degree_type):
    result_dict = dict()
    # check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    # set logging
    # I guess there exists conflict. So add code below.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_filepath = outpath + label + '_' + degree_type + '_powerlaw.log'
    logging.basicConfig(filename=log_filepath, format='%(asctime)s - %(levelname)s - %('
                                                      'message)s', level=logging.INFO)
    data = []
    trueid_degree_records = np.genfromtxt(filename)
    for item in trueid_degree_records:
        data.append(item[1])
    fit = powerlaw.Fit(data, discrete=True)
    result_dict[degree_type + '-deg-powerlaw-xmin'] = fit.xmin
    result_dict[degree_type + '-deg-powerlaw-alpha'] = fit.power_law.alpha
    logging.info('power law fit result:')
    logging.info('Is discrete: {0}'.format(fit.estimate_discrete))
    logging.info('x_min = {0}'.format(fit.xmin))
    logging.info('power_law_alpha = {0}'.format(fit.power_law.alpha))
    logging.info('power_law_D = {0}'.format(fit.power_law.D))
    logging.info('\n')

    R1, p1 = fit.distribution_compare('power_law', 'exponential')
    logging.info('Compare with exponential:')
    logging.info('R = {0}'.format(R1))
    logging.info('p = {0}'.format(p1))
    logging.info('\n')

    R2, p2 = fit.distribution_compare('power_law', 'truncated_power_law')
    logging.info('Compare with truncated power law:')
    logging.info('R = {0}'.format(R2))
    logging.info('p = {0}'.format(p2))
    logging.info('\n')

    R3, p3 = fit.distribution_compare('power_law', 'lognormal')
    logging.info('Compare with lognormal:')
    logging.info('R = {0}'.format(R3))
    logging.info('p = {0}'.format(p3))
    logging.info('\n')

    R4, p4 = fit.distribution_compare('power_law', 'stretched_exponential')
    logging.info('Compare with stretched exponential:')
    logging.info('R = {0}'.format(R4))
    logging.info('p = {0}'.format(p4))
    logging.info('\n')

    logging.info('Truncated power law fit result: ')
    logging.info('{0} = {1}'.format(fit.truncated_power_law.parameter1_name, fit.truncated_power_law.parameter1))
    logging.info('{0} = {1}'.format(fit.truncated_power_law.parameter2_name, fit.truncated_power_law.parameter2))
    logging.info('\n')

    logging.info('Lognormal fit result: ')
    logging.info('{0} = {1}'.format(fit.lognormal.parameter1_name, fit.lognormal.parameter1))
    logging.info('{0} = {1}'.format(fit.lognormal.parameter2_name, fit.lognormal.parameter2))

    ####
    color_type = {'all': 'b', 'in': 'g', 'out': 'c', 'reciprocity': 'r'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fit.plot_ccdf(color=color_type[degree_type], linewidth=2, ax=ax)
    fit.power_law.plot_ccdf(color=color_type[degree_type], linestyle='--', ax=ax)
    ####
    ax.set_ylabel(u"p(X≥x)")
    ax.set_xlabel(u"x")

    figname = outpath + label + '-' + degree_type + '-powerlaw'
    plt.savefig(figname + '.eps', bbox_inches='tight')
    return result_dict
