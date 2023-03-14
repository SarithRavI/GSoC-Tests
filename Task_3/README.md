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

![alt text](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_3/Resources/ml4sci-gsoc-gnn-architecture.png?raw=true)

#### **2.1. Notes on implemented models**
- `N=0` (i.e no pre-processing layers), `L=2` & `M=2`
- Two models are implemented with two types of GNN layers: (1) [GCN Layer](https://arxiv.org/pdf/1609.02907.pdf) (2) [PointNet Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PointNetConv.html#torch_geometric.nn.conv.PointNetConv). 
- Latent embedding dimension is 300 for all the GNN layers. Embedding dimension of post-processing MLPs is 300 except for last MLP where embedding dimension = `number of classes = 2`.
- Dropout ratio is set to 0.3 in both models.
- Skip-connections are implemented for each GNN layer.
- Both `sum` & `last` Jumping Knowledge connections are implemented. `last` JK-connection is in use. Note `last` JK only pass the node representation embedding of the last layer to next module, while `sum` JK passes summation of node representation embedding of every layer.  Refer to section 5.3.3 of [this](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf) for the specification of JK.
- `sum`, `mean`, `max` & `attention` graph pooling is implemented. currently `mean` pooling is used.
- In both models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 32`, `epcohs : 75`.
- In addition to aforementioned GNN layers, we trained models with [GAT layer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) & [GIN layer](https://arxiv.org/pdf/1810.00826.pdf), and found that aforementioned layers outperform these.

### 3. Performance

- Training of the model was done by splitting the dataset into train: validation: test sets with 70%, 20%, 10% ratio respectively.
- Performance metric is AUC (of ROC).
- nb inx denotes notebook cell index to refer the model training. 

| Layer | has GPE | Train  | Test  | nb inx |
| ------ | :---: | :----: | :----: | :----: |
| PointNet Conv | no| 0.793 | 0.773 | In [19]
| PointNet Conv | yes | - | - | -
| GCN | no | 0.791 | 0.777| In [28]
| GCN | yes | 0.784 | 0.768 | Removed

#### 3.1 Discussion

- Model that utilizes PointNet conv outperforms the model with GCN layers in training. But GCN model has higher generalization capability. This observation aligns with the fact that simple convolution can increase the linear seperability of a model and improves generalization (see this [paper](https://arxiv.org/pdf/2102.06966.pdf)).
- GCN layer is sensitive to distribution of node features. When graphs falling into separate classes, have less amount of similar node features among each other, GCN is powerful as much as WL-test. This explains why GCN outperformed GIN model in our case.(see this [paper](https://arxiv.org/pdf/1810.00826.pdf))
- Knowing that, since all the nodes have a similar element (GPE value along depth dim which is fixed to 0.0) in their node feature, we suspect this element reduces performance of the GCN when dataset with GPE is fed. This hypothesis is yet to be checked.

### 4. Future works.

- [ ] Trying different approaches to convert images to graphs. specially in a way such that the graph is small.
- [ ] Finding the best `k` value for creating k-nearest-neighbor graph.
- [ ] Finding where the model best fits in the GNN design space (see this [paper](https://arxiv.org/pdf/2011.08843.pdf))
- [ ] Trying graph coarsening approaches in place of graph pooling (see page 65 of [here])(https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
