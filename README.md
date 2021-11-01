# TAKN
Dataset and code of  paper "Temporal Analysis of Knowledge Networks" (ICBK'2021)

## Dataset

We have released [64 topic networks](https://zenodo.org/record/5602570) extracted from the English Wikipedia's internal link network. Every topic network contains 19 snapshots, which are taken on March 1st of each year from 2002 to 2020. For example, for Mathematics, we have

    ```
    enwiki-2002-03-01-Mathematics-edgelist-simple.tsv   39.7 kB
    enwiki-2003-03-01-Mathematics-edgelist-simple.tsv   127.0 kB
    enwiki-2004-03-01-Mathematics-edgelist-simple.tsv   380.6 kB
    enwiki-2005-03-01-Mathematics-edgelist-simple.tsv   794.8 kB
    enwiki-2006-03-01-Mathematics-edgelist-simple.tsv   1.3 MB
    enwiki-2007-03-01-Mathematics-edgelist-simple.tsv   1.8 MB
    enwiki-2008-03-01-Mathematics-edgelist-simple.tsv   2.3 MB
    enwiki-2009-03-01-Mathematics-edgelist-simple.tsv   2.8 MB
    enwiki-2010-03-01-Mathematics-edgelist-simple.tsv   3.3 MB
    enwiki-2011-03-01-Mathematics-edgelist-simple.tsv   3.6 MB
    enwiki-2012-03-01-Mathematics-edgelist-simple.tsv   4.0 MB
    enwiki-2013-03-01-Mathematics-edgelist-simple.tsv   4.3 MB
    enwiki-2014-03-01-Mathematics-edgelist-simple.tsv   4.6 MB
    enwiki-2015-03-01-Mathematics-edgelist-simple.tsv   5.0 MB
    enwiki-2016-03-01-Mathematics-edgelist-simple.tsv   5.2 MB
    enwiki-2017-03-01-Mathematics-edgelist-simple.tsv   5.4 MB
    enwiki-2018-03-01-Mathematics-edgelist-simple.tsv   5.7 MB
    enwiki-2019-03-01-Mathematics-edgelist-simple.tsv   5.9 MB
    enwiki-2020-03-01-Mathematics-edgelist-simple.tsv   6.1 MB 
    ```

Data format of *enwiki-YEAR-03-01-TOPIC-edgelist-simple.tsv*: There are two columns separated with tab. The first column is the page id of the source page, and the second column is the page id of the target page. Below is a part of *enwiki-2020-03-01-Mathematics-edgelist-simple.tsv*:

```
# source_page_id    target_page_id
9815    98759
338403  901459
15397886        29965
2333880 61478
...
```

All 64 topic networks are available at: [https://zenodo.org/record/5602570](https://zenodo.org/record/5602570).

If you want to generate topic networks by yourself, here are steps:

1. Download [Wikipedia dump](https://dumps.wikimedia.org/enwiki/).

1. Produce the WikiLinkGraphs following [Cristian Consonni et al.](https://ojs.aaai.org/index.php/ICWSM/article/view/3257). We removed redirects from the WikiLinkGraphs.

    Sample:
    ```
    # source_page_id    target_page_id
    1   2
    1   3
    2   4
    ...
    ```

1. Classify each Wikipedia page into relevant topics following [Isaac Johnson et al.](https://dl.acm.org/doi/abs/10.1145/3442442.3452347?casa_token=igZ5MeIls-MAAAAA:ogc6WmjHFT-nj6y5w3Z7J3wTGX5nJgPNziFNe8mJG6OPLkAiDIssKcFaZYNnd-juqWD7GPfhc5oriQ). For example:

    ```
    # page_id:   topics
    1: Biology, History
    2: Mathematics, Biography, Asia
    3: Literature
    4: Technology
    ...
    ```

1. Extract topic networks from the WikiLinkGraphs based on the topic classification of Wikipedia pages.

## Network Analysis

Requirements:
- powerlaw
- matplotlib
- pandas
- networkit==8.0
- numpy
- scipy

```python
python -m network_scanner
```

## Citation

```bibtex
@inproceedings{consonni2019wikilinkgraphs,
  title={WikiLinkGraphs: a complete, longitudinal and multi-language dataset of the Wikipedia link networks},
  author={Consonni, Cristian and Laniado, David and Montresor, Alberto},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={13},
  pages={598--607},
  year={2019}
}
```

```
@inproceedings{johnson2021language,
  title={Language-agnostic Topic Classification for Wikipedia},
  author={Johnson, Isaac and Gerlach, Martin and S{\'a}ez-Trumper, Diego},
  booktitle={Companion Proceedings of the Web Conference 2021},
  pages={594--601},
  year={2021}
}
```

```bibtex
@inproceedings{huang2021temporal,
  title={Temporal Analysis of Knowledge Networks},
  author={Huang, Xikun and Wang, Chuanqing and Sun, Qilin and Li, Yangyang and Li, Weizhuo},
  booktitle={2021 IEEE International Conference on Big Knowledge},
  year={2021}
}
```
