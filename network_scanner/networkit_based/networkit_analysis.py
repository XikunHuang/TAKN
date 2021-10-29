import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('../')
import logging
import networkit as nk
import os
import numpy as np

import network_scanner.networkit_based.networkit_util as networkit_util
import network_scanner.networkit_based.networkit_plot as networkit_plot
import network_scanner.networkit_based.degree_analysis as degree_analysis
import network_scanner.networkit_based.components_analysis as components_analysis


def analysis_undirected(net, label, outpath):
    """
    Analyze undirected network
    Args:
        net: networkit graph object
        label: network name
        outpath: result path

    Returns:

    """
    # Store result in dict
    result_dict = dict()
    # Check whether the graph is undirected
    is_directed = net.isDirected()
    if is_directed:
        logging.error('Input graph should be undirected')
    else:
        logging.info('Undirected graph')

    # Check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Profiling
    nodes = net.numberOfNodes()
    edges = net.numberOfEdges()
    result_dict['u_num_of_nodes'] = nodes # u means undirected
    result_dict['u_num_of_edges'] = edges
    logging.info('Number of nodes: {0}'.format(nodes))
    logging.info('Number of edges: {0}'.format(edges))
    pf = nk.profiling.Profile.create(net, preset="minimal")
    pf.output("HTML", outpath)
    # os.rename(outpath+label+'.html', outpath+label+'-undirected.html')
    logging.info('\n')

    # Degree distribution
    logging.info('Write degree to file...')
    networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='all')
    logging.info('Start to plot degree distribution...')
    uniqe_deg_seq = networkit_plot.plot_degree_dist(net, label, outpath)
    min_deg = min(uniqe_deg_seq)
    max_deg = max(uniqe_deg_seq)
    avg_deg = 2*edges/nodes
    result_dict['u_min_deg'] = min_deg
    result_dict['u_max_deg'] = max_deg
    result_dict['u_avg_deg'] = avg_deg
    logging.info('Min degree: {0}'.format(min_deg))
    logging.info('Max degree: {0}'.format(max_deg))
    logging.info('Average degree: {0}'.format(avg_deg))
    logging.info('\n')

    # CCDF
    logging.info('Start to plot complementary cumulative (in/out) degree distribution...')
    networkit_plot.plot_ccum_degree_dist(net, label, outpath, degree_type='all')
    logging.info('Plot cc degree distribution done.\n')

    # Powerlaw
    degree_analysis.get_deg_seq(net, label, outpath, degree_type='all')
    degree_seq_filename = outpath + label + '-all-degree'
    degree_analysis.power_law_analysis(degree_seq_filename, label, outpath, degree_type='all')

    # Clustering coefficient
    # Global clustering coefficient. The first definition in http://konect.uni-koblenz.de/statistics/clusco
    logging.info('Calculating clustering coefficient...')
    global_cc = nk.globals.ClusteringCoefficient.approxGlobal(net, nodes)
    result_dict['appox_clustering_coefficient'] = global_cc
    logging.info('u_approximate clustering coefficient: {0}'.format(global_cc))
    logging.info('')
    logging.info('Plot cumulative distribution of local clustering coefficient...')
    networkit_plot.plot_cum_clustering_dist(net, label, outpath, turbo=True)
    logging.info('Plot local clustering coefficient done.\n')
    # The second definition in http://konect.uni-koblenz.de/statistics/clusco.
    # cc = globals.clustering(net, error=0.01)
    # print("clustering coefficient ", cc)

    # Connected components
    logging.info('Plot connected components...')
    connected_components = networkit_plot.plot_connected_component_dist(net, label, outpath)
    logging.info('Number of connected components: {0}'.format(len(connected_components)))
    lcc = np.max(connected_components)
    logging.info('The largest connected component size: {0}'.format(lcc))
    lcc_subgraph = networkit_util.get_lcc_subgraph(net)
    lcc_nodes_per = lcc_subgraph.numberOfNodes() / net.numberOfNodes()
    result_dict['u_lwcc_nodes_percentage'] = lcc_nodes_per
    logging.info('LCC nodes percentage: {0}'.format(lcc_nodes_per))
    logging.info('LCC edges percentage: {0}'.format(lcc_subgraph.numberOfEdges() / net.numberOfEdges()))
    logging.info('Components done.\n')

    # Eigenvector centrality
    logging.info('Calculating eigenvector centrality...')
    networkit_plot.plot_eigenvector_centrality(net, label, outpath)
    centrality_name = 'eigenvector-centrality'
    centrality_filename = outpath + label + '-' + centrality_name + '-falseid-value'
    paras = {'centrality_filename': centrality_filename, 'label': label, 'outpath': outpath,
             'centrality_name': centrality_name}
    networkit_plot.plot_ccum_centrality_dist(**paras)
    logging.info('Eigenvector centrality done.\n')

    # Distance
    logging.info('Calculating diameter...')
    diameter = nk.distance.Diameter(net)
    diameter.run()
    result_dict['u_diameter'] = diameter.getDiameter()
    logging.info('Diameter: {0}'.format(diameter.getDiameter()))
    logging.info('\n')
    # Currently only undirected connected graph is supported for effective diameter
    logging.info('Calculating effective diameter...')
    eff_diameter = nk.distance.EffectiveDiameterApproximation(lcc_subgraph, ratio=0.9)
    eff_diameter.run()
    result_dict['u_effective_diameter'] = eff_diameter.getEffectiveDiameter()
    logging.info('Effective diameter: {0}'.format(eff_diameter.getEffectiveDiameter()))
    logging.info('Distance done.\n')

    # Assorsativity
    logging.info('Calculating assorsativity...')
    assorsativity = networkit_util.get_assorsativity(net)
    result_dict['u_assorsativity'] = assorsativity
    logging.info('Assorsativity: {0}'.format(assorsativity))
    logging.info('Plot assorsativity...')
    networkit_plot.plot_assorsativity(net, label, outpath, degree_type='all')
    logging.info('Assorsativity done\n')
    return result_dict


def analysis_directed(net, label, outpath):
    """
    Analyze directed network.
    """
    result_dict = dict()
    # Check whether graph is directed
    is_directed = net.isDirected()
    if not is_directed:
        logging.error('Input graph should be directed.')
    else:
        logging.info('Directed graph')

    # Check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Profiling
    nodes = net.numberOfNodes()
    edges = net.numberOfEdges()
    result_dict['d_num_of_nodes'] = nodes
    result_dict['d_num_of_edges'] = edges
    logging.info('Number of nodes: {0}'.format(nodes))
    logging.info('Number of edges: {0}'.format(edges))
    pf = nk.profiling.Profile.create(net, preset="minimal")
    pf.output("HTML", outpath)
    # os.rename(outpath+label+'.html', outpath+label+'-directed.html')
    logging.info('\n')

    # In-degree distribution
    logging.info('Write indegree to file...')
    networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='in')
    logging.info('Start to plot in-degree distribution...')
    uniqe_deg_seq = networkit_plot.plot_indeg_dist(net, label, outpath)
    min_indeg = min(uniqe_deg_seq)
    max_indeg = max(uniqe_deg_seq)
    result_dict['d_min_indeg'] = min_indeg
    result_dict['d_max_indeg'] = max_indeg
    logging.info('Min in-degree: {0}'.format(min_indeg))
    logging.info('Max in-degree: {0}'.format(max_indeg))
    logging.info('\n')

    # Out-degree distribution
    logging.info('Write outdegree to file...')
    networkit_util.get_and_write_deg_dist(net, label, outpath, degree_type='out')
    logging.info('Start to plot out-degree distribution...')
    uniqe_deg_seq = networkit_plot.plot_outdeg_dist(net, label, outpath)
    min_outdeg = min(uniqe_deg_seq)
    max_outdeg = max(uniqe_deg_seq)
    result_dict['d_min_outdeg'] = min_indeg
    result_dict['d_max_outdeg'] = max_outdeg
    logging.info('Min out-degree: {0}'.format(min_indeg))
    logging.info('Max out-degree: {0}'.format(max_outdeg))
    logging.info('\n')

    # CCDF
    logging.info('Start to plot complementary cumulative (in/out) degree distribution...')
    networkit_plot.plot_ccum_degree_dist(net, label, outpath, degree_type='in')
    networkit_plot.plot_ccum_degree_dist(net, label, outpath, degree_type='out')
    logging.info('Plot cc (in/out) degree distribution done.\n')

    # In-Out-degree
    logging.info('Plot outdegree vs indegree...')
    networkit_plot.plot_out_in_degree_comparision(net, label, outpath)
    logging.info('Plot out vs in done.\n')

    # Powerlaw
    degree_analysis.get_deg_seq(net, label, outpath, degree_type='in')
    degree_seq_filename = outpath + label + '-in-degree'
    degree_analysis.power_law_analysis(degree_seq_filename, label, outpath, degree_type='in')
    degree_analysis.get_deg_seq(net, label, outpath, degree_type='out')
    degree_seq_filename = outpath + label + '-out-degree'
    degree_analysis.power_law_analysis(degree_seq_filename, label, outpath, degree_type='out')

    # Reciprocity
    logging.info('Calculating reciprocity...')
    reciprocity = networkit_util.get_reciprocity(net)
    result_dict['d_reciprocity'] = reciprocity
    logging.info('Reciprocity: {0}'.format(reciprocity))
    logging.info('Reciprocity done.\n')

    # Connected components
    # Weakly connected components
    logging.info('Plot wcc distribution...')
    wcc = networkit_plot.plot_wcc_dist(net, label, outpath)
    logging.info('Number of weakly connected components: {0}'.format(len(wcc)))
    lwcc = np.max(wcc)
    logging.info('The largest weakly connected component size: {0}'.format(lwcc))
    logging.info('')
    # Strongly connected components
    logging.info('Plot scc distribution...')
    scc = networkit_plot.plot_scc_dist(net, label, outpath)
    logging.info('Number of strongly connected components: {0}'.format(len(scc)))
    lscc = np.max(scc)
    logging.info('The largest strongly connected component size: {0}'.format(lscc))
    lscc = networkit_util.get_lscc_subgraph(net)
    lscc_nodes_per = lscc.numberOfNodes() / net.numberOfNodes()
    result_dict['d_lscc_nodes_percentage'] = lscc_nodes_per
    logging.info('LCC nodes percentage: {0}'.format(lscc_nodes_per))
    logging.info('LCC edges percentage: {0}'.format(lscc.numberOfEdges() / net.numberOfEdges()))
    # Macro structure
    components_analysis.run(net, label, outpath)
    logging.info('Components done.\n')

    # Pagerank
    logging.info('Calculating pagerank...')
    networkit_plot.plot_pagerank(net, label, outpath)
    centrality_name = 'pagerank'
    centrality_filename = outpath + label + '-' + centrality_name + '-falseid-value'
    paras = {'centrality_filename': centrality_filename, 'label': label, 'outpath': outpath,
             'centrality_name': centrality_name}
    networkit_plot.plot_ccum_centrality_dist(**paras)
    logging.info('Pagerank done.\n')
    return result_dict


def analysis(filepath, label, outpath, directed=False):
    """
    Network analysis.
    Args:
        filepath:
        label:
        outpath:
        directed:treat the network as undirected (False) or directed (True).

    Returns:

    """
    # Check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    if not directed:
        # Set logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_filepath = outpath + label + '-undirected.log'
        logging.basicConfig(filename=log_filepath,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        logging.info('Read graph from file')
        # Read network
        separator = '\t'
        firstNode = 0
        continuous = False
        directed = False
        net_reader = nk.graphio.EdgeListReader(separator=separator,
                                               firstNode=firstNode,
                                               continuous=continuous,
                                               directed=directed)
        net = net_reader.read(filepath)
        logging.info('Get undirected map node id...')
        networkit_util.write_map_node_id(net_reader, label, outpath + 'undirected-')
        logging.info('******************************************')
        paras = {'net': net, 'label': label, 'outpath': outpath}
        result_dict = analysis_undirected(**paras)
        logging.info('******************************************')
        logging.info('Done ^-^')
        return result_dict
    if directed:
        # Set logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_filepath = outpath + label + '-directed.log'
        logging.basicConfig(filename=log_filepath,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        # Read network
        separator = '\t'
        firstNode = 0
        continuous = False
        directed = True
        net_reader = nk.graphio.EdgeListReader(separator=separator,
                                               firstNode=firstNode,
                                               continuous=continuous,
                                               directed=directed)
        net = net_reader.read(filepath)
        logging.info('Get directed map node id...')
        networkit_util.write_map_node_id(net_reader, label, outpath + 'directed-')
        paras = {'net': net, 'label': label, 'outpath': outpath}
        result_dict = analysis_directed(**paras)
        logging.info('Done ^-^')
        return result_dict