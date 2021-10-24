import numpy as np
import scipy as sp
import random

graphon_vertical_divider = 0.5
graphon_horizontal_divider = 0.5
graphon_p = 0.2
graphon_q = 0.8

def generateData(max_alternatives):

    route_label_dict = {}
    num_alternatives = random.randint(2, max_alternatives)
    route_features = np.zeros((num_alternatives, 3))
    correlation_matrix = np.zeros((num_alternatives, num_alternatives))
    truth_values = np.zeros((num_alternatives, 1))

    for i in range(num_alternatives):
        a_rand = random.uniform()
        b_rand = random.uniform()
        c_rand = random.uniform()
        route_label = random.uniform()

        route_features[i] = np.array([a_rand, b_rand, c_rand])
        route_label_dict.update({i:route_label})

        b_eq = 0
        e_eq = 0

        for j in range(i):
            corr_route_label = route_label_dict.get(j)
            
            if ((route_label < graphon_vertical_divider and corr_route_label < graphon_horizontal_divider) 
                or (route_label >= graphon_vertical_divider and corr_route_label >= graphon_horizontal_divider)):
                
                correlation_matrix[i][j] = graphon_p

                b_eq = 5 + (3 * graphon_p + random.uniform()
                    + 3 * (1 - graphon_p) + random.uniform()) + random.uniform()
                e_eq = (graphon_p + random.uniform()
                    + (1 - graphon_p) + random.uniform()) + random.uniform()
            else:
                correlation_matrix[i][j] = graphon_q

                b_eq = 5 + (3 * graphon_p + random.uniform() 
                    + 3 * (graphon_q - graphon_p) + random.uniform()
                    + 3 * (1 - graphon_q) + random.uniform()) + random.uniform()
                e_eq = (graphon_p + random.uniform()
                    + (graphon_q - graphon_p) + random.uniform()
                    + (1 - graphon_q)) + random.uniform() 

        h_eq = random.uniform()
        g_eq = random.uniform()
        d_eq = 2 * h_eq + random.uniform() 
        c_eq = g_eq + random.uniform()
        a_eq = random.uniform()
        beta_a = random.uniform()
        beta_b = random.uniform()
        beta_c_d = random.uniform()
        beta_e = random.uniform()
        v_eq = beta_a * a_eq + beta_b * b_eq + beta_c_d * c_eq * d_eq + beta_e * e_eq
        u_eq = v_eq + random.uniform()
        truth_values[i] = u_eq

    prob_y = np.exp(truth_values) / np.sum(np.exp(truth_values))
    index_y = np.random.choice(num_alternatives, p=prob_y)
    y = np.zeros(num_alternatives)
    y[index_y] = 1

    return route_features, correlation_matrix, y