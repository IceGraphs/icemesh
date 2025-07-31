import h5py
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
import torch
from torch_geometric.data import Data
from icemesh import utils
from pathlib import Path


def convert_to_torch(datafile, preprocessed_data, number_trajectories, number_ts, dt):

    with h5py.File(datafile, 'r') as data:
        #Define the list that will return the data graphs
        data_list = []

        if not preprocessed_data.is_file():
            for i,trajectory in enumerate(data.keys()):
                if(i==number_trajectories):
                    break
                print("Trajectory: ",i)

                #We iterate over all the time steps to produce an example graph except
                #for the last one, which does not have a following time step to produce
                #node output values
                for ts in range(len(data[trajectory]['velocity'])-1):

                    if(ts==number_ts):
                        break

                    #Get node features

                    #Note that it's faster to convert to numpy then to torch than to
                    #import to torch from h5 format directly
                    momentum = torch.tensor(np.array(data[trajectory]['velocity'][ts]))
                    node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), utils.NodeType.SIZE))).squeeze(1)
                    x = torch.cat((momentum,node_type),dim=-1).type(torch.float)

                    #Get edge indices in COO format
                    edges = utils.triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

                    edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                                torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

                    #Get edge features
                    u_i=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[0]]
                    u_j=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[1]]
                    u_ij=u_i-u_j
                    u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
                    edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

                    #Node outputs, for training (velocity)
                    v_t=torch.tensor(np.array(data[trajectory]['velocity'][ts]))
                    v_tp1=torch.tensor(np.array(data[trajectory]['velocity'][ts+1]))
                    y=((v_tp1-v_t)/dt).type(torch.float)

                    #Node outputs, for testing integrator (pressure)
                    p=torch.tensor(np.array(data[trajectory]['pressure'][ts]))

                    #Data needed for visualization code
                    cells=torch.tensor(np.array(data[trajectory]['cells'][ts]))
                    mesh_pos=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))

                    data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,p=p,
                                        cells=cells,mesh_pos=mesh_pos))

                    torch.save(data_list, preprocessed_data)
                    print(f"Output Name: {preprocessed_data}")

        else:
            print(f"preprocessed dataset found at {preprocessed_data}")
            dataset = torch.load(preprocessed_data, weights_only=False)

    return dataset