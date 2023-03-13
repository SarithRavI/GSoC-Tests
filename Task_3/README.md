## GNN based jets classification.

Method of graph construction from images.

- Single image is a `125x125x3 matrix`. This can be viewed as `125x125 pixels with 3 channel values`.
- Filter pixels where the sum along 3 channels (i.e sum along the depth dimension) is greater than 0.0
- These filtered pixels are treated as nodes of the graph.
- Global position embedding (GPE) of the node is coordinates of the pixel along height,width & depth dimension, considering the graph is inside 3D point cloud. Note the value of depth (i.e Z) dimention is fixed to 0.0 since the graph is a 2D facet inside 3D point cloud.  
- In total a node has 6 features (= channel values (3) + GPE (3) ). GPE is optional and can drop before feeding the graph to the GNN model.
- To produce edge connections connect each node with it's `k nearest neighbors`. Metric of distance computation is euclidean distance. In this implementation `k=2`.
- Edge feature corresponds to euclidean distance between pair of interconnected node.
