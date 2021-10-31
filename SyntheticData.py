import numpy as np
import scipy as sp
import random

def generateData(max_alternatives, beta_a, beta_b, beta_c_d, beta_e, graphon_vertical_boundary=0.5, graphon_horizontal_boundary=0.5, graphon_p=0.2, graphon_q=0.8, debug=False):

    route_label_dict = {}
    num_alternatives = random.randint(2, max(max_alternatives, 2))
    if debug:
        print("Num alternatives:", num_alternatives)
    route_features = np.zeros((num_alternatives, 3))
    correlation_matrix = np.zeros((num_alternatives, num_alternatives))
    truth_values = np.zeros((num_alternatives))

    for i in range(num_alternatives):
        route_label = random.uniform(0, 1)
        route_label_dict.update({i:route_label})

        if debug:
            print("============================================")
            print("Iteration:", i)
            print("-------------------")

        b_eq = 0
        e_eq = 0

        for j in range(i):
            corr_route_label = route_label_dict.get(j)
            
            if ((route_label < graphon_vertical_boundary and corr_route_label < graphon_horizontal_boundary) 
                or (route_label >= graphon_vertical_boundary and corr_route_label >= graphon_horizontal_boundary)):
                
                correlation_matrix[i][j] = graphon_p
                correlation_matrix[j][i] = graphon_p

                b_eq = 5 + (3 * graphon_p + random.uniform(0, 1)
                    + 3 * (1 - graphon_p) + random.uniform(0, 1)) + random.uniform(0, 1)
                e_eq = (graphon_p + random.uniform(0, 1)
                    + (1 - graphon_p) + random.uniform(0, 1)) + random.uniform(0, 1)
            else:
                correlation_matrix[i][j] = graphon_q
                correlation_matrix[j][i] = graphon_q

                b_eq = 5 + (3 * graphon_p + random.uniform(0, 1) 
                    + 3 * (graphon_q - graphon_p) + random.uniform(0, 1)
                    + 3 * (1 - graphon_q) + random.uniform(0, 1)) + random.uniform(0, 1)
                e_eq = (graphon_p + random.uniform(0, 1)
                    + (graphon_q - graphon_p) + random.uniform(0, 1)
                    + (1 - graphon_q)) + random.uniform(0, 1) 

        h_eq = random.uniform(0, 1)
        g_eq = random.uniform(0, 1)
        d_eq = 2 * h_eq + random.uniform(0, 1) 
        c_eq = g_eq + random.uniform(0, 1)
        a_eq = random.uniform(0, 1)
        v_eq = beta_a * a_eq + beta_b * b_eq + beta_c_d * c_eq * d_eq + beta_e * e_eq
        u_eq = v_eq + random.uniform(0, 1)
        truth_values[i] = u_eq

        route_features[i] = np.array([a_eq, b_eq, c_eq])
        if debug:
            print("Truth equation values:")
            print("a:", a_eq)
            print("b:", b_eq)
            print("c:", c_eq)
            print("d:", d_eq)
            print("e:", e_eq)
            print("g:", g_eq)
            print("h:", h_eq)
            print("v:", v_eq)
            print("u:", u_eq)

    prob_y = np.exp(truth_values) / np.sum(np.exp(truth_values))
    index_y = np.random.choice(num_alternatives, p=prob_y)
    y = np.zeros(num_alternatives)
    y[index_y] = 1

    return route_features, correlation_matrix, y

def main():
    route_features, corr, y = generateData(5, 1, 1, 1, 1, debug=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(route_features)
    print(corr)
    print(y)

if __name__ == "__main__":
    main()