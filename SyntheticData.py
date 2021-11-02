import numpy as np
import scipy as sp
import random

def get_corr_value(x, y, graphon_vertical, graphon_horizontal, graphon_p, graphon_q):
    """ Helper method for generate data that retrieves the correlation value between two alternative routes"""

    if (x < graphon_vertical and y < graphon_horizontal) or (x >= graphon_vertical and y >= graphon_horizontal):
        return graphon_p
    else:
        return graphon_q


def generateData(max_alternatives, beta_a, beta_b, beta_c_d, beta_e, graphon_vertical_boundary=0.5, graphon_horizontal_boundary=0.5, graphon_p=0.2, graphon_q=0.8, debug=False):
    """ Generates synthetic data for a single actor

            Keyword arguments:
                max_alternatives (int): maximum alternative routes to be generated, randomly picked between [2,max_alternatives] (must be greater than 2 to have effects)
                beta_a (float): value of bet_a in the truth equation
                beta_b (float): value of beta_b in the truth equation
                beta_c_d (float): value of beta_c_d in the truth equation
                beta_e (float): value of beta_e in the truth equation
                graphon_vertical_boundary (float): value of n in x=n, the line that determines the vertical boundary of the graphon used to determine route correlation; defaults to 0.5
                graphon_horizontal_boundary (float): value of n in y=n, the line that determines the horizontal boundary of the graphon used to determine route correlation; defaults to 0.5
                graphon_p (float): sets the p value of the graphon used to determine route correlation; must abide by p <= 1.0; defaults to 0.2
                graphon_q (float): sets the q value of the graphon used to determine route correlation; must abide by q <= 1.0; defaults to 0.8
                debug (bool): prints out debug information; defaults to False

            returns:
                route_features: a num_alternatives x 3 matrix containing the alternative routes and their features (num_alternatives is chosen randomly from [2, max_alternatives])
                correlation_matrix: the correlation matrix for the alternative routes
                y: the y matrix, shows which route the actor chose

    """
    route_labels = []
    max_corr = []
    num_alternatives = random.randint(2, max(max_alternatives, 2))
    if debug:
        print("Num alternatives:", num_alternatives)

    # Initializing the return values
    route_features = np.zeros((num_alternatives, 3))
    correlation_matrix = np.zeros((num_alternatives, num_alternatives))
    truth_values = np.zeros((num_alternatives))

    # Iterate through all alternative route pairs and determine their correlation, adding it to the correlation matrix
    # Also determine the maximum correlation a route experiences, used to determine the number of route sections it has
    for alt in range(num_alternatives):
        label = random.uniform(0, 1)
        route_labels.append(label)
        max_corr.append(0)

        for prev_alt in range(alt):
            corr_val = get_corr_value(route_labels[alt], route_labels[prev_alt], graphon_vertical_boundary, graphon_horizontal_boundary, graphon_p, graphon_q)
            correlation_matrix[alt][prev_alt] = corr_val
            correlation_matrix[prev_alt][alt] = corr_val
            
            max_corr[prev_alt] = max(max_corr[prev_alt], corr_val)
            max_corr[alt] = max(max_corr[alt], corr_val)

    if debug:
        print("route_labels:", route_labels)
        print("max_corr:", max_corr)

    # Iterate through all routes and perform calculations of the truth equation on each
    for alt in range(num_alternatives):

        if debug:
            print("==============================================")
            print("Alternate route #:", alt)
            print("----------------------------------------------")            

        b_eq = 0
        e_eq = 0
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
        truth_values[alt] = u_eq

        route_features[alt] = np.array([a_eq, b_eq, c_eq])

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
       
    # Perform a softmax over the utility values
    prob_y = np.exp(truth_values) / np.sum(np.exp(truth_values))
    index_y = np.random.choice(num_alternatives, p=prob_y)
    y = np.zeros(num_alternatives)
    y[index_y] = 1

    return route_features, correlation_matrix, y

def main():
    route_features, corr, y = generateData(5, 1, 1, 1, 1, debug=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Route Features:")
    print(route_features)
    print("Correlation matrix:")
    print(corr)
    print("Y matrix:")
    print(y)

if __name__ == "__main__":
    main()