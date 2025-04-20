# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp

def define_MDP():
    #*****************define MDP*****************#
    # define states
    grid_rows, grid_cols = 3, 4
    coordinate_index_mapping = {}
    index_coordinate_mapping = {}
    cnt = 0
    for i in range(1, grid_rows+1):
        for j in range(1, grid_cols+1):
            if i==2 and j==2:
                pass
            else:
                coordinate_index_mapping[(i,j)] = cnt
                index_coordinate_mapping[cnt] = (i,j)
                cnt += 1
    states_coordinate = list(coordinate_index_mapping.keys())
    states = list(index_coordinate_mapping.keys())
    state_goal = coordinate_index_mapping[(3,4)]
    state_tiger = coordinate_index_mapping[(2,4)]
    # print(states)

    # define actions
    actions = [0, 1, 2, 3]
    index_action_mapping = {0: "N", 1 : "S", 2: "E", 3 : "W"}
    pos_deltas = {"N": (1,0), "S": (-1,0), "E": (0,1), "W": (0,-1)}
    # print(actions)

    # define Transition probability
    trans_probs = np.zeros((len(states),len(actions),len(states)))
    transitions = {"N": {"N": 0.8,"W": 0.1,"E": 0.1 }, "S":{"S": 0.8, "W": 0.1, "E": 0.1}, "E":{"E": 0.8, "N": 0.1, "S": 0.1}, "W":{"W": 0.8,"N": 0.1, "S": 0.1}}
    #Transition probability for non-absorbing states
    for s in states:
        if s != state_goal and s != state_tiger:
            for a in actions:
                transition = transitions[index_action_mapping[a]]
                for direction in transition.keys():
                    current_coordinate = index_coordinate_mapping[s]
                    pos_delt = pos_deltas[direction]
                    next_coordinate = tuple(a + b for a, b in zip(current_coordinate, pos_delt))
                    if next_coordinate not in states_coordinate:
                        next_coordinate = current_coordinate   # new pos if within grid and not wall
                    next_state = coordinate_index_mapping[next_coordinate]
                    trans_probs[s,a,next_state] += transition[direction]
    #Transition probability for absorbing states
    trans_probs[state_goal][:, state_goal] = np.ones(4)
    trans_probs[state_tiger][:, state_tiger] = np.ones(4)
    # print(trans_probs)

    # define reward function
    reward = np.zeros((len(states),len(actions),len(states)))
    for s in states :
        for a in actions:
            for next_s in states:
                if next_s == coordinate_index_mapping[(2,4)]:
                    if next_s == s:
                        reward[s,a,next_s] = 0
                    else:
                        reward[s,a,next_s] = -1
                elif next_s == coordinate_index_mapping[(3,4)]:
                    if next_s == s:
                        reward[s,a,next_s] = 0
                    else:
                        reward[s,a,next_s] = +1
                else:
                    reward[s,a,next_s] = -0.25

    # define initial state distribution
    b_0 = [1/ len(states)] * len(states)
    # print(b_0)

    #discount factor gamma
    gamma = 0.99

    return states, index_coordinate_mapping, actions, index_action_mapping, trans_probs, reward, b_0, gamma

def solve_MDP_LP_primal(states, actions, trans_probs, reward, b_0, gamma):
    V = cp.Variable(len(states)) # Variables V(s)
    objective = cp.Minimize(np.asarray(b_0) @ V) # Problem Objective
    constraints = [] # List of Constraints
    for s in states:
        for a in actions:
            constraints += [V[s] >= trans_probs[s,a] @ reward[s,a] + gamma * (trans_probs[s,a] @ V)]
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve()
    prob.solve(solver = cp.GLPK, verbose = False)

    # Deterministic policy pi(s)
    pi = [None] * len(states)

    # pi(s) = action a which maximizes Q(s, a)
    for s in states:
        bestQ = float('-inf')
        for a in actions:
            currQ = trans_probs[s,a] @ reward[s,a] + gamma * (trans_probs[s,a] @ V).value
            if currQ > bestQ:
                bestQ = currQ
                pi[s] = a
    return prob.value, V.value, pi

def solve_MDP_LP_dual(states, actions, trans_probs, reward, b_0, gamma):
    # TODO:create Variables
    V = cp.Variable()
    x = cp.Variable((len(states), len(actions)), nonneg=True) # Variables x(s,a)
    r_sa = np.zeros((len(states), len(actions)))
    for s in states:
        for a in actions:
            r_sa[s, a] = trans_probs[s, a, :] @ reward[s, a, :]

    # objective function = maximize x(s,a)*reward(s,a)
    objective = cp.Maximize(cp.sum(cp.multiply(x, r_sa)))
    constraints = []
    # Constraints
    # flow constraint sum(x(s',a')) = b_0(s') + gamma * sum(x(s,a))
    for sprime in states:
        lhs = cp.sum(x[sprime, :])  # sum_{a'} x(s',a')
        # sum_{s,a} P(s'|s,a)* x(s,a):
        flow_in = cp.sum(cp.multiply(trans_probs[:, :, sprime], x))
        rhs = b_0[sprime] + gamma * flow_in
        constraints.append(lhs == rhs)

    # TODO:solve the dual LP problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GLPK, verbose=False)

    # TODO: extract deterministic policy pi(s)
    x_opt = x.value
    pi = [None] * len(states)
    pi = [np.argmax(x_opt[s, :]) for s in states]

    return prob.value, V.value, pi

def print_optimal_policy(states, states_pos_mapping, actions_direction_mapping, pi):
    coordinate_index_mapping = {}
    for key in states_pos_mapping.keys():
        coordinate_index_mapping[states_pos_mapping[key]] = key
    for i in reversed(range(1, 3+1)):   # Print like a "mathematical graph"
        out_str = ""
        for j in range(1, 4+1):
            pos = (i, j)
            if i==3 and j==4:
                s = "Goal"
            elif i==2 and j==4:
                s = "Tiger"
            elif i==2 and j==2:
                s = ""
            else:
                state = coordinate_index_mapping[pos]
                s = str(actions_direction_mapping[pi[state]])
            out_str += s.ljust(10)
        print(out_str)
    print("\n")

def main():
    #Define a MDP.
    #Please do not change this function.
    #States, actions, and b_0 are returned as a list. Trans_probs(s,a,s') and reward(s,a,s') are returned as a matrix.
    states, states_pos_mapping, actions, actions_direction_mapping, trans_probs, reward, b_0, gamma = define_MDP()

    #Please write your own code to implement  dual LP
    #primal_optimal_policy and dual_optimal_policy should indicate the best action to take in each state.
    #Please return the optimal policy as a list of action index to be printed out using the print_optimal_policy function
    #for example optimal_policy = [0,0,0,0,0,0,0,0,0,0,0]
    primal_objective, primal_variables, primal_optimal_policy = solve_MDP_LP_primal(states, actions, trans_probs, reward, b_0, gamma)
    dual_objective, dual_variables, dual_optimal_policy = solve_MDP_LP_dual(states, actions, trans_probs, reward, b_0, gamma)

    #print function
    print("*******results from solving primal LP*******\n")
    print("objective value of primal LP: {}\n".format(primal_objective))
    print("optimal policy from primal LP:\n")
    print_optimal_policy(states, states_pos_mapping, actions_direction_mapping, primal_optimal_policy)

    print("*******results from solving dual LP*******\n")
    print("objective value of dual LP: {}\n".format(dual_objective))
    print("optimal policy from dual LP:\n")
    print_optimal_policy(states, states_pos_mapping, actions_direction_mapping, dual_optimal_policy)

    print("Dual gap: ", primal_objective - dual_objective)

if __name__ == "__main__":
    main()