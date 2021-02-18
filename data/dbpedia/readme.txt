DBPedia Ontology Classification Dataset

Version 2, Updated 09/09/2015


LICENSE

The DBpedia datasets are licensed under the terms of the Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. For more information, please refer to http://dbpedia.org. For a recent overview paper about DBpedia, please refer to: Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, Sören Auer, Christian Bizer: DBpedia – A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia. Semantic Web Journal, Vol. 6 No. 2, pp 167–195, 2015.

The DBPedia ontology classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu), licensed under the terms of the Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. It is used as a text classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).


DESCRIPTION

The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 14), title and content. The title and content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). There are no new lines in title or content.
