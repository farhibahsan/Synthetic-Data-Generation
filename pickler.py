import pickle
import torch
from SyntheticData import *
from torch_geometric.data import Data

if __name__ == "__main__":
    num_actors = 1000
    max_alternatives = 10
    beta_a = 1
    beta_b = 1
    beta_c_d  = 1
    beta_e = 1
    graphon_vertical_boundary = 0.5
    graphon_horizontal_boundary = 0.5
    graphon_p = 0.2
    graphon_q = 0.8

    data_list = []

    for actor in range(num_actors):
        x, y, edge_attr, edge_index = generateData(max_alternatives, beta_a, beta_b, beta_c_d, beta_e, 
            graphon_vertical_boundary=graphon_vertical_boundary, graphon_horizontal_boundary=graphon_horizontal_boundary, 
            graphon_p=graphon_p, graphon_q=graphon_q, gpu=True)

        data_list.append(Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index))
    
    outfile = open('syntheticdata', 'wb')
    pickle.dump(data_list, outfile)
    outfile.close()

    print("Done pickling")
