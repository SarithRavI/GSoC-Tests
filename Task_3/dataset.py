import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix
import pyarrow.parquet as pq

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

def split(data, batch):
    """
    PyG util code to create graph batches
    """
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    # Edge indices should start at zero for every graph.
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': []}
    if data.x is not None:
        slices['x'] = node_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices

def read_graph_data(path):
    
    dataset = pq.read_table(path,columns=["X_jets","y"]).to_pandas()
    images_raw = dataset["X_jets"]
    labels = dataset["y"]
    
    num_node_attr = 6
    num_edge_attr =1 

    x = np.empty([0,num_node_attr],dtype=np.float32)
    edge_index = np.empty([2,0],dtype=np.int32)
    edge_attr = np.empty([0,num_edge_attr],dtype=np.float32)
    node_graph_id=np.array([],dtype=np.int32)
    edge_slice = np.array([0],dtype=np.int32)
    y = np.empty([0,1],dtype=np.float32)

    for img_inx,img in enumerate(tqdm(images_raw)):
        # convert the images into np arrays of shape (3,125,125)
        num_dim = img.shape[0]
        img_np = np.stack([np.stack(channel) for channel in img])
        # get 2D mask of values 
        # where values are pixel values across the depth/ channel dim
        mask = np.sum(img_np, axis=0)
        # get x,y coordinates of this mask of non zero values 
        mask_coord = np.nonzero(mask)
        # global position embeddings 
        # here global positions are in 3D dim
        global_locs = np.vstack((mask_coord[0],mask_coord[1],np.zeros(mask_coord[0].shape,dtype=np.int32))).T
        # get vertices for the graph  
        # vertices shape is (3, # num of nodes)
        vertices=img_np[:,mask_coord[0],mask_coord[1]].T.astype(np.float32) 
        # node features with global position embeddings
        print(vertices.dtype)
        vertices = np.hstack((vertices,global_locs)).astype(np.float32) 
        print(vertices.dtype)
        print(global_locs.dtype)
        
        # self loops are excluded
        # adj : adjacency matrix of the image
        adj = kneighbors_graph(vertices, 2, mode='connectivity', include_self=False)
        vertices_dist = kneighbors_graph(vertices, 2, mode='distance', include_self=False)
        img_edge_index  = from_scipy_sparse_matrix(adj)[0].numpy().astype(np.int32)
        img_edge_index_ls= img_edge_index.tolist()
        
        img_edge_attr = vertices_dist[img_edge_index_ls[0],
                                                   img_edge_index_ls[1]].reshape(-1,1).astype(np.int32)
        
        x = np.vstack((x,vertices))
        num_nodes = vertices.shape[0] # number of nodes 
        # append node_graph_id 
        graph_inx = img_inx
        node_graph_id = np.append(node_graph_id,[graph_inx]*num_nodes)
        # add image edge_index
        edge_index = np.hstack((edge_index,img_edge_index))
        # add image edge attr
        edge_attr = np.vstack((edge_attr,img_edge_attr))
                  
        edge_slice = np.append(edge_slice,edge_slice[-1]+img_edge_index[0].shape[-1])
        y = np.vstack((y,labels[graph_inx].reshape(1,-1)))

    # converting to torch tensors 
    edge_index = torch.from_numpy(edge_index).to(torch.int64)
    edge_attr = torch.from_numpy(edge_attr).to(torch.float32)
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    node_graph_id = torch.from_numpy(node_graph_id)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, node_graph_id)

    slices['edge_index']=torch.from_numpy(edge_slice).to(torch.int32)
    if data.edge_attr is not None:
        slices['edge_attr'] = torch.from_numpy(edge_slice).to(torch.int32)
    
    return data, slices

class JetsGraphsDataset(InMemoryDataset):

    def __init__(self,root,name,transform=None, pre_transform=None,pre_filter=None):
        self.name = name
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(JetsGraphsDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def download(self):
        pass
    @property
    def raw_dir(self):
        name = "raw/"
        return osp.join(self.root,self.name,name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root,self.name,name)

    @property
    def num_node_attributes(self):
        pass 
    
    
    def num_landmarks(self):
        return self.data.y.size(1)

    @property
    def raw_file_names(self):
        names =['QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet',
                'QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet',
                'QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet']
        return names

    @property
    def processed_file_names(self):
        return 'geometric_jets_processed.pt'

    def process(self):
        raw_file = [file for file in self.raw_file_names if self.name in file][0]
        self.data,self.slices = read_graph_data(osp.join(self.raw_dir,raw_file))
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)
            
        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
            
        if self.transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
            
        print('Saving...')
        torch.save((self.data, self.slices), self.processed_paths[0])
        