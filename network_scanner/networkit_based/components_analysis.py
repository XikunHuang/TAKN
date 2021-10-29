import logging
import os
import queue
import numpy as np
import networkit as nk


def get_scc(net, label, outpath):
    """
    Calculate strongly connected components.
    Args:
        net:
        label:
        outpath:

    Returns:

    """
    # Check whether the output directory exists
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Set logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_filepath = outpath + label + '-scc.log'
    logging.basicConfig(filename=log_filepath,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info('Start to calculate...')
    scc = nk.components.StronglyConnectedComponents(net)
    scc.run()
    num_of_scc = scc.numberOfComponents()
    logging.info('Number of scc components: {0}'.format(num_of_scc))
    scc_partitions = scc.getPartition()
    # reference https://networkit.iti.kit.edu/api/networkit.html?highlight=partition#networkit.Partition
    # Get scc sizes
    scc_sizes = scc_partitions.subsetSizes()
    scc_sizes_file = outpath + label + '-scc-sizes.csv'
    with open(scc_sizes_file, 'w') as f:
        for item in scc_sizes:
            f.write(str(item) + '\n')
    # Get the largest scc
    index_size_map = scc_partitions.subsetSizeMap()
    lscc_index = 0
    lscc_size = 0
    for k, v in index_size_map.items():
        if v > lscc_size:
            lscc_index = k
            lscc_size = v
    logging.info('LSCC size: {0}'.format(lscc_size))
    lscc = scc_partitions.getMembers(lscc_index)
    lscc_file = outpath + label + '-LSCC-falseid.csv'
    with open(lscc_file, 'w') as f:
        for item in lscc:
            f.write(str(item) + '\n')
    return scc_partitions, lscc_index


def child_nodes_bfs(net, source):
    """
    Get child nodes of given node using BFS.
    Args:
        net:
        source:id of the given node

    Returns:
        list of nodes that can be reached by source.
    """
    child_nodes = []
    n = net.upperNodeIdBound()
    visited = np.zeros(n)
    q = queue.Queue()
    q.put(source)
    visited[source] = 1
    while not q.empty():
        # Get and remove the 1st item
        u = q.get()
        for v in net.iterNeighbors(u):
            if visited[v] == 0:
                q.put(v)
                visited[v] = 1
    for item in net.iterNodes():
        if visited[item] == 1:
            child_nodes.append(item)
    return child_nodes


def parent_nodes_bfs(net, target):
    """
    Get parent nodes of given node using BFS.
    Args:
        net:
        target:

    Returns:
        list of nodes that can reach target.
    """
    parent_nodes = []
    n = net.upperNodeIdBound()
    visited = np.zeros(n)
    q = queue.Queue()
    q.put(target)
    visited[target] = 1
    while not q.empty():
        u = q.get()
        for v in net.iterInNeighbors(u):
            if visited[v] == 0:
                q.put(v)
                visited[v] = 1
    for item in net.iterNodes():
        if visited[item] == 1:
            parent_nodes.append(item)
    return parent_nodes


def has_path(net, source, target):
    """
    Calculate whether there is a path from source to target.
    Args:
        net:
        source: node id used in networkit, not raw id in edgelist file.
        target: node id used in networkit, not raw id in edgelist file.

    Returns:

    """
    n = net.upperNodeIdBound()
    if target > n:
        return False
    visited = np.zeros(n)
    q = queue.Queue()
    q.put(source)
    visited[source] = 1
    while not q.empty():
        u = q.get()
        for v in net.iterNeighbors(u):
            if visited[v] == 0:
                if target == v:
                    return True
                q.put(v)
                visited[v] = 1
    return False


def get_in_and_out(net, label, outpath):
    """
    Get IN and OUT components.
    """
    # get LSCC
    lscc_set = set()
    source = 0
    target = 0
    lscc_filename = outpath + label + '-LSCC-falseid.csv'
    with open(lscc_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            id = int(line.strip())
            lscc_set.add(id)
        source = id
        target = id
    in_and_lscc = parent_nodes_bfs(net, target)
    out_and_lscc = child_nodes_bfs(net, source)
    # Get IN
    in_part = []
    for item in in_and_lscc:
        if item not in lscc_set:
            in_part.append(item)
    # Get OUT
    out_part = []
    for item in out_and_lscc:
        if item not in lscc_set:
            out_part.append(item)
    # Write to file
    in_part_filename = outpath + label + '-IN-falsid.csv'
    with open(in_part_filename, 'w') as f:
        for item in in_part:
            f.write(str(item) + '\n')
    out_part_filename = outpath + label + '-OUT-falseid.csv'
    with open(out_part_filename, 'w') as f:
        for item in out_part:
            f.write(str(item) + '\n') 


def get_each_components(net, scc_partitions, lscc_index, label, outpath):
    """
    Analyze the macro structure.
    """
    # Set logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_filepath = outpath + label + '-topology-structure.log'
    logging.basicConfig(filename=log_filepath,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info(label + ': ')
    N = net.numberOfNodes()
    # Get LSCC
    lscc_set = scc_partitions.getMembers(lscc_index)
    lscc_prop = len(lscc_set) / (N + 0.0)
    logging.info('LSCC size: {0}'.format(lscc_prop))
    # Get IN
    in_part_set = set()
    in_part_filename = outpath + label + '-IN-falsid.csv'
    with open(in_part_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            id = int(line.strip())
            in_part_set.add(id)
    in_part_prop = len(in_part_set) / (N + 0.0)
    logging.info('IN_size: {0}'.format(in_part_prop))
    # Get OUT
    out_part_set = set()
    out_part_filename = outpath + label + '-OUT-falseid.csv'
    with open(out_part_filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            id = int(line.strip())
            out_part_set.add(id)
    out_part_prop = len(out_part_set) / (N + 0.0)
    logging.info('OUT_size: {0}'.format(out_part_prop))
    # Get In-TENDRILS and OUT-TENDRILS
    in_tendrils_part = []
    out_tendrils_part = []
    for id in scc_partitions.getSubsetIds():
        # Check whether current scc is LSCC
        if id == lscc_index:
            continue
        cur_scc_set = scc_partitions.getMembers(id)
        # Actually, we just want a random node in cur_scc_set
        for cur_node in cur_scc_set:
            # Check whether current scc belongs to IN or OUT
            if cur_node in in_part_set or cur_node in out_part_set:
                continue
            # Can current scc be reached from IN?
            parents_of_cur_nodes = parent_nodes_bfs(net, cur_node)
            parents_set_of_cur_nodes = set(parents_of_cur_nodes)
            for in_item in in_part_set:
                if in_item in parents_set_of_cur_nodes:
                    for cur_item in cur_scc_set:
                        in_tendrils_part.append(cur_item)
                    break
            # Can current scc reach OUT?
            children_of_cur_nodes = child_nodes_bfs(net, cur_node)
            children_set_of_cur_nodes = set(children_of_cur_nodes)
            for out_item in out_part_set:
                if out_item in children_set_of_cur_nodes:
                    for cur_item in cur_scc_set:
                        out_tendrils_part.append(cur_item)
                    break
            # WE JUST WANT A RANDOM NODE
            break
    # Get TUBE
    out_tendrils_set = set(out_tendrils_part)
    tube_size = 0
    for item in in_tendrils_part:
        if item in out_tendrils_set:
            tube_size += 1
    in_tendrils_prop = (len(in_tendrils_part) - tube_size) / (N + 0.0)
    logging.info('IN-TENDRILS_size: {0}'.format(in_tendrils_prop))
    out_tendrils_prop = (len(out_tendrils_part) - tube_size) / (N + 0.0)
    logging.info('OUT-TENDRILS_size: {0}'.format(out_tendrils_prop))
    logging.info('TENDRILS_size: {0}'.format(in_tendrils_prop + out_tendrils_prop))
    tube_prop = tube_size / (N + 0.0)
    logging.info('TUBE_size: {0}\n'.format(tube_prop))


def run(net, label, outpath):
    scc_partitions, lscc_index = get_scc(net, label, outpath)
    get_in_and_out(net, label, outpath)
    get_each_components(net, scc_partitions, lscc_index, label, outpath)
