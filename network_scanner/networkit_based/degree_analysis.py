import numpy as np
import os
import powerlaw
from matplotlib import pyplot as plt
import logging
import network_scanner.networkit_based.networkit_util as networkit_util


def get_deg_seq(net, label, outpath, degree_type):
    """
    Get and store degree sequence
    Args:
        net:
        label:
        outpath:
        degree_type: [all, in, out]

    Returns:

    """
    assert degree_type in ['all', 'in', 'out']
    outfile = open(outpath + label + '-' + degree_type + '-degree', 'w')
    ret = []
    if degree_type == 'all':
        ret = [net.degree(node) for node in net.iterNodes()]
    if degree_type == 'in':
        ret = [net.degreeIn(node) for node in net.iterNodes()]
    if degree_type == 'out':
        ret = [net.degreeOut(node) for node in net.iterNodes()]
    for ele in ret:
        outfile.write(str(ele) +'\n')
    outfile.close()


def power_law_analysis(filename, label, outpath, degree_type):
    """
    Fit degree distribution.
    Args:
        filename:
        label:
        outpath:
        degree_type: [all, in, out]

    Returns:

    """
    result_dict = dict()
    # Check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    # Set logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_filepath = outpath + label + '_' + degree_type + '_powerlaw.log'
    logging.basicConfig(filename=log_filepath,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    data = np.genfromtxt(filename)
    fit = powerlaw.Fit(data, discrete=True)
    result_dict[degree_type + '-deg-powerlaw-xmin'] = fit.xmin
    result_dict[degree_type + '-deg-powerlaw-alpha'] = fit.power_law.alpha
    logging.info('Power law fit result:')
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

    color_type = {'all': 'b', 'in': 'g', 'out': 'c'}
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fit.plot_ccdf(color=color_type[degree_type], linewidth=2, ax=ax)
    fit.power_law.plot_ccdf(color=color_type[degree_type], linestyle='--', ax=ax)
    ax.set_ylabel(u"p(X≥x)")
    ax.set_xlabel(u"x")

    figname = outpath + label + '-powerlaw'
    plt.savefig(figname + '.eps', bbox_inches='tight')
    return result_dict


def get_falseid_value_map(false_id_value_filename):
    """
    Get falseid_value_map from file
    Args:
        false_id_value_filename: the first column is false id, the second column is value

    Returns:

    """
    false_id_value_file = open(false_id_value_filename, 'r')
    false_id_value_map = dict()
    while True:
        line = false_id_value_file.readline()
        if not line:
            break
        split_line = line.strip().split(sep='\t')
        false_id_value_map[split_line[0]] = split_line[1]
    false_id_value_file.close()
    return false_id_value_map


def get_falseid_name_map(map_filename, index_filename):
    """
    Get falseid-name map. Key is falseid, and value is name.
    """
    map_file = open(map_filename, 'r')
    falseid_trueid_map = dict()
    while True:
        line = map_file.readline()
        if not line:
            break
        split_line = line.strip().split(sep='\t')
        falseid_trueid_map[split_line[1]] = split_line[0]
    map_file.close()
    index_file = open(index_filename, 'r')
    trueid_name_map = dict()
    while True:
        line = index_file.readline()
        if not line:
            break
        split_line = line.strip().split(sep='\t')
        trueid_name_map[split_line[0]] = split_line[1]
    index_file.close()
    falseid_name_map = dict()
    for falseid in falseid_trueid_map.keys():
        if falseid_trueid_map[falseid] in trueid_name_map:
            falseid_name_map[falseid] = trueid_name_map[falseid_trueid_map[falseid]]
        else:
            print(falseid_trueid_map[falseid])
    return falseid_name_map


def get_top_k_trueid_and_name(false_id_value_filename, map_filename, index_filename, result_filename, k):
    """
    Get top k value and name.
    """
    false_id_value_map = get_falseid_value_map(false_id_value_filename)
    sorted_false_id_value_pair = sorted(false_id_value_map.items(), key=lambda ele: int(ele[1]), reverse=True)
    top_k_falseid, top_k_value = zip(*sorted_false_id_value_pair[0:k])
    top_k_trueid = networkit_util.falseid_to_trueid(top_k_falseid, map_filename)
    top_k_name = networkit_util.trueid_to_name(top_k_trueid, index_filename)
    result_file = open(result_filename, 'w')
    for i in range(len(top_k_name)):
        result_file.write(str(top_k_value[i]) + '\t' + str(top_k_name[i]) + '\n')
    result_file.close()


def get_top_in_out_degree_cmp(indeg_falseid_value_filename, outdeg_falseid_value_filename, map_filename,
                              index_filename, result_filename):
    """
    Analyze indegree vs outdegree
    """
    indeg_falseid_value_map = get_falseid_value_map(indeg_falseid_value_filename)
    outdeg_falseid_value_map = get_falseid_value_map(outdeg_falseid_value_filename)
    in_close_out = []
    in_larger_out = []
    out_larger_in = []
    for false_id in indeg_falseid_value_map.keys():
        in_minus_out = int(indeg_falseid_value_map[false_id]) - int(outdeg_falseid_value_map[false_id])
        out_minus_in = int(outdeg_falseid_value_map[false_id]) - int(indeg_falseid_value_map[false_id])
        if in_minus_out < 10 and out_minus_in < 10:
            in_close_out.append((false_id, abs(in_minus_out), int(indeg_falseid_value_map[false_id]), int(outdeg_falseid_value_map[false_id])))
        if in_minus_out > 0:
            in_larger_out.append((false_id, in_minus_out, int(indeg_falseid_value_map[false_id]), int(outdeg_falseid_value_map[false_id])))
        if out_minus_in > 0:
            out_larger_in.append((false_id, out_minus_in, int(indeg_falseid_value_map[false_id]), int(outdeg_falseid_value_map[false_id])))
    sorted_in_close_out = sorted(in_close_out, key=lambda ele: int(ele[2]), reverse=True)[0:100]
    sorted_in_larger_out = sorted(in_larger_out, key=lambda ele: int(ele[1]), reverse=True)[0:100]
    sorted_out_larger_in = sorted(out_larger_in, key=lambda ele: int(ele[1]), reverse=True)[0:100]
    in_close_out = []
    in_larger_out = []
    out_larger_in = []
    falseid_name_map = get_falseid_name_map(map_filename, index_filename)
    for ele in sorted_in_close_out:
        if ele[0] in falseid_name_map.keys():
            in_close_out.append((falseid_name_map[ele[0]], ele[1], ele[2], ele[3]))
        else:
            print(ele[0])
    for ele in sorted_in_larger_out:
        if ele[0] in falseid_name_map.keys():
            in_larger_out.append((falseid_name_map[ele[0]], ele[1], ele[2], ele[3]))
        else:
            print(ele[0])
    for ele in sorted_out_larger_in:
        if ele[0] in falseid_name_map.keys():
            out_larger_in.append((falseid_name_map[ele[0]], ele[1], ele[2], ele[3]))
        else:
            print(ele[0])
    result_file = open(result_filename, 'w')
    result_file.write("format(name, result, indegree, outdegree) \n")
    result_file.write("Indegree is close to outdegree(<10): \n")
    result_file.write('\n')
    for ele in in_close_out:
        result_file.write(str(ele))
        result_file.write('\n')
    result_file.write('\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('\n')
    result_file.write("Indegree is larger than outdegree: \n")
    result_file.write('\n')
    for ele in in_larger_out:
        result_file.write(str(ele))
        result_file.write('\n')
    result_file.write('\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('*' * 40 + '\n')
    result_file.write('\n')
    result_file.write("Outdegree is larger than indegree: \n")
    result_file.write('\n')
    for ele in out_larger_in:
        result_file.write(str(ele))
        result_file.write('\n')
    result_file.close()
