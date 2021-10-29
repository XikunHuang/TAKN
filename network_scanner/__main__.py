import network_scanner.networkit_based.networkit_analysis as network_analysis


def main():
    filepath = './input/enwiki-2020-03-01-Mathematics-edgelist-simple.tsv'
    # filepath = './input/toy_graph.tsv'
    label = 'math'
    # label = 'toy'
    outpath = './output/' + label + '/'
    # Undirected
    network_analysis.analysis(filepath, label, outpath, directed=False)
    # Dircted
    network_analysis.analysis(filepath, label, outpath, directed=True)


if __name__ == '__main__':
    main()