import h5py
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
import torch
from torch_geometric.data import Data
from icemesh import utils


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

def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list):
    """Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    """
    #mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    #mean and std of the output parameters are calculated
    mean_vec_y=torch.zeros(data_list[0].y.shape[1:])
    std_vec_y=torch.zeros(data_list[0].y.shape[1:])

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6

    #Define a very small value for normalizing to
    eps=torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0

    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        mean_vec_y+=torch.sum(dp.y,dim=0)
        std_vec_y+=torch.sum(dp.y**2,dim=0)
        num_accs_y+=dp.y.shape[0]

        if(num_accs_x>max_accumulations or num_accs_edge>max_accumulations or num_accs_y>max_accumulations):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)

    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]

    return mean_std_list