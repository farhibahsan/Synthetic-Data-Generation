import pickle
from SyntheticData import *

class Data(object):
    def __init__(self, x, y, edge_attr, edge_index):
        super(Data, self).__init__()
        self._x = x
        self._y = y
        self._edge_attr = edge_attr
        self._edge_index = edge_index

    @property
    def x(self):
        return self.x

    @property
    def y(self):
        return self.y

    @property
    def edge_attr(self):
        return self.edge_attr

    @property
    def edge_index(self):
        return self.edge_index

if __name__ == "__main__":
    num_actors = 20
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
        route_features, correlation_matrix, y = generateData(max_alternatives, beta_a, beta_b, beta_c_d, beta_e, 
            graphon_vertical_boundary=graphon_vertical_boundary, graphon_horizontal_boundary=graphon_horizontal_boundary, 
            graphon_p=graphon_p, graphon_q=graphon_q)
        
        edge_matrix = np.ones((2, int(route_features.shape[1] * (route_features.shape[1] - 1) / 2)))

        data_list.append(Data(route_features, y, edge_matrix, correlation_matrix))
    
    outfile = open('syntheticdata', 'wb')
    pickle.dump(data_list, outfile)
    outfile.close()
