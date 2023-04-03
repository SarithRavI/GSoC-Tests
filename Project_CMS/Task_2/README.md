## GNN based jets classification.

**_SPECIAL NOTE:_** 
**_Due to a thermal issue in my personal machine my machine tends to turn off unexpectedly while training. After each epoch we save the model weights as a checkpoint So we can continue the training from that checkpoint if the machine turned off amid training. Please note that's why there are notebook outputs with epochs starting from where it left off (you can see the logic in cell `In [12]`)._** 

**_So if you want to view the training outcome at the end of each model's training, please refer to following cells:_**
- **_GNN with PointNet Conv: `In [13]`_**
- **_GNN with GCN Layer: `In [14]`_**
- **_GNN with GraphGPS Layer: `In [15]`_**


### **1. Method of graph construction from images.**

**_NOTE: We created image dataset only with raw file identified by `QG_jets_1` in [here](https://zenodo.org/record/3164691#.ZCrKXHZBxEa) and all 100000 images of this have been used._**

- Dataset construction logic is in [dataset.py](https://github.com/SarithRavI/GSoC-Tests/blob/master/Project_CMS/Task_2/dataset.py) file
- Single image is a `1x134x4 matrix`. This can be viewed as `1x134x4 pixels with 4 channel values`.
- Filter pixels where the sum along 4 channels (i.e sum along the depth dimension) is greater than 0.0
- These filtered pixels are treated as nodes of the graph.
- Global position embedding (GPE) of the node is coordinates of the pixel along height & width dimension, considering the graph is inside 3D point cloud. Note the value of depth (i.e Z) dimention is fixed to 0.0 since the graph is a 2D facet inside 3D point cloud, hence we do not consider Z dimension in GPE.  
- Since the height is always 1, the value of height coord is always 0.0.
- In total a node has 6 features (= channel values (4) + GPE (2 i.e x,y) ). GPE is optional and can drop before feeding the graph to the GNN model.
- To produce edge connections connect each node with it's `k nearest neighbors`. Metric of distance computation is euclidean distance. In this implementation `k=4`.
- Edge feature corresponds to euclidean distance between pair of interconnected nodes.

### **2. Model Architecture**

![alt text](https://github.com/SarithRavI/ML4SCI-GSoC-Tests/blob/test/Task_3/Resources/ml4sci-gsoc-gnn-architecture.png?raw=true)

#### **2.1. Notes on implemented models**
- Three models are implemented with two types of GNN layers: (1) [GCN Layer](https://arxiv.org/pdf/1609.02907.pdf) (2) [PointNet Conv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PointNetConv.html#torch_geometric.nn.conv.PointNetConv) and (3)[GraphGPS](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GPSConv.html#torch_geometric.nn.conv.GPSConv). 
- In case where PointConv and GCN being used `N=0` (i.e no pre-processing layers), and in case GraphGPS being used `N=1`.
- For all models `L=2` & `M=2`
- Latent embedding dimension is 300 for all the GNN layers. Embedding dimension of post-processing MLPs is 300 except for last MLP where embedding dimension = `number of classes = 2`.
- Dropout ratio is set to 0.3 in all models.
- Skip-connections are implemented for each GNN layer.
- Both `sum` & `last` Jumping Knowledge connections are implemented. `last` JK-connection is in use. Note `last` JK only pass the node representation embedding of the last layer to next module, while `sum` JK passes summation of node representation embedding of every layer.  Refer to section 5.3.3 of [this](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf) for the specification of JK.
- `sum`, `mean`, `max` & `attention` graph pooling is implemented. currently `mean` pooling is used.
- In all models, `optimizer : Adam`, `learning rate : 1e-3`, `batch size : 32`.
- All the models are trained for `75` epochs.
- In addition to aforementioned GNN layers, we implemented models with [GAT layer](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) & [GIN layer](https://arxiv.org/pdf/1810.00826.pdf).

### 3. Performance

- Training of the model was done by splitting the dataset into train: validation: test sets with 70%, 20%, 10% ratio respectively.
- Performance metric is AUC (of ROC).
- nb inx denotes notebook cell index to refer the model training. 

| Layer | has GPE | Train  | Test  | nb inx |
| ------ | :---: | :----: | :----: | :----: |
| PointNet Conv | no| 0.877 | 0.870 | In [13]
| GCN | yes (only xy) | 0.875 | 0.873 | In [14]
| GraphGPS | yes (+ *RW SPE) | 0.862 | 0.862| In [15]

**_*Here we denote positional encoding like Random-walk (RW) embeddings as structural postional encoding i.e, SPE._** 

#### 3.1 Discussion

- Model that utilizes PointNet conv outperforms the model with GCN layers and GraphGPS in training. But GCN model has higher generalization capability. This observation aligns with the fact that simple convolution can increase the linear seperability of a model and improves generalization (see this [paper](https://arxiv.org/pdf/2102.06966.pdf)).
- Model trained with GraphGPS layer has similar performance in both train and test data. This is due to the fact that transformers can capture more long-distance features beyond the common inductive bias in graphs i.e local structural features. 

### 4. Future works.
- [x] Supporting structural positional encoding like Random-walk embedding.
- [ ] Finding the best `k` value for creating k-nearest-neighbor graph.
- [ ] Finding where the model best fits in the GNN design space (see this [paper](https://arxiv.org/pdf/2011.08843.pdf))
- [ ] Trying graph coarsening approaches in place of graph pooling (see page 65 of [here](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf))
