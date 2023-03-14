## GNN based jets classification.

**_SPECIAL NOTE:_** 
**_Due to a thermal issue in my personal machine my machine tends to turn off unexpectedly while training. After each epoch we save the model weights as a checkpoint So we can continue the training from that checkpoint if the machine turned off amid training. Please note that's why there are several notebook outputs for a single model training each starting from the epoch where it left off (you can see the logic in cell `In [12]`)._** 

**_So if you want to view the training outcome at the end of each model's training, please refer to following cells:_**
- **_GNN with PointNet Conv: `In [19]`_**
- **_GNN with GCN Layer: `In [28]`_**


### **1. Method of graph construction from images.**

- Dataset construction logic is in [dataset.py](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_3/dataset.py) file
- Single image is a `125x125x3 matrix`. This can be viewed as `125x125 pixels with 3 channel values`.
- Filter pixels where the sum along 3 channels (i.e sum along the depth dimension) is greater than 0.0
- These filtered pixels are treated as nodes of the graph.
- Global position embedding (GPE) of the node is coordinates of the pixel along height,width & depth dimension, considering the graph is inside 3D point cloud. Note the value of depth (i.e Z) dimention is fixed to 0.0 since the graph is a 2D facet inside 3D point cloud.  
- In total a node has 6 features (= channel values (3) + GPE (3) ). GPE is optional and can drop before feeding the graph to the GNN model.
- To produce edge connections connect each node with it's `k nearest neighbors`. Metric of distance computation is euclidean distance. In this implementation `k=2`.
- Edge feature corresponds to euclidean distance between pair of interconnected node.

### **2. Model Architecture**

![alt text](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_3/Resources/ml4sci-gsoc-gnn-archi.jpg?raw=true)

#### **2.1. Notes on implemented models**
- `N=0` (i.e no pre-processing layers), `L=2` & `M=2`
- Two models are implemented with two types of GNN layers: (1) [GCN Layer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv) (2) [PointNet Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PointNetConv.html#torch_geometric.nn.conv.PointNetConv). 
- Latent embedding dimension is 300 for all the GNN layers. Embedding dimension of post-processing MLPs is 300 except for last MLP where embedding dimension = `number of classes = 2`.
- Dropout ratio is set to 0.3 in both models.
- Skip-connections are implemented for each GNN layer.
- Both `sum` & `last` Jumping Knowledge connections are implemented. `last` JK-connection is in use. Note `last` JK only pass the node representation embedding of the last layer to next module, while `sum` JK passes summation of node representation embedding of every layer.  Refer to section 5.3.3 of [this](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf) for the specification of JK.
- In both models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 32`, `epcohs : 75`.

