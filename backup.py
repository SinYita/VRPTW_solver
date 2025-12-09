import os
#Please change the location of license accordingly
os.environ['GRB_LICENSE_FILE'] = '/Users/weiyuandu/gurobi.lic'
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import math
import argparse
from gurobipy import *
from itertools import product
from Visualizer import visualize_vrptw_instance


class Data:
    customerNum = 0
    nodeNum = 0
    vehicleNum = 0
    capacity = 0
    corX = []
    corY = []
    demand = []
    serviceTime = []
    readyTime = []
    dueTime = []
    distanceMatrix = [[]]


def readData(path, customerNum):
    data = Data()
    data.customerNum = customerNum
    if customerNum is not None:
        data.nodeNum = customerNum + 2
    with open(path, 'r') as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            count += 1
            if count == 5:
                line = line[:-1]
                s = re.split(r" +", line)
                data.vehicleNum = int(s[1])
                data.capacity = float(s[2])
            elif count >= 10 and (customerNum is None or count <= 10 + customerNum):
                line = line[:-1]
                s = re.split(r" +", line)
                data.corX.append(float(s[2]))
                data.corY.append(float(s[3]))
                data.demand.append(float(s[4]))
                data.readyTime.append(float(s[5]))
                data.dueTime.append(float(s[6]))
                data.serviceTime.append(float(s[7]))
    data.nodeNum = len(data.corX) + 1
    data.customerNum = data.nodeNum - 2
    # 回路
    data.corX.append(data.corX[0])
    data.corY.append(data.corY[0])
    data.demand.append(data.demand[0])
    data.readyTime.append(data.readyTime[0])
    data.dueTime.append(data.dueTime[0])
    data.serviceTime.append(data.serviceTime[0])
    # 计算距离矩阵
    data.distanceMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(data.nodeNum):
        for j in range(i + 1, data.nodeNum):
            distance = math.sqrt((data.corX[i] - data.corX[j]) ** 2 + (data.corY[i] - data.corY[j]) ** 2)
            data.distanceMatrix[i][j] = data.distanceMatrix[j][i] = distance
    return data


class Solution:
    ObjVal = 0
    X = [[]]
    routes = [[]]
    routeNum = 0

    def __init__(self, data, model):
        self.ObjVal = model.ObjVal
        # X_ijk
        self.X = [[[0 for k in range(data.vehicleNum)] for j in range(data.nodeNum)] for i in range(data.nodeNum)]
        # routes
        self.routes = []


def getSolution(data, model):
    solution = Solution(data, model)
    for v in model.getVars():
        split_arr = re.split(r'[,\[\]]', v.VarName)
        if split_arr[0] == 'x' and v.x != 0:
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = v.x
        elif split_arr[0] == 's':
            pass
    
    for k in range(data.vehicleNum):
        i = 0
        subRoute = [i]
        finish = False
        while not finish:
            for j in range(data.nodeNum):
                if solution.X[i][j][k] != 0:
                    subRoute.append(j)
                    i = j
                    if j == data.nodeNum - 1:
                        finish = True
                        break
        if len(subRoute) >= 3:
            subRoute[-1] = 0
            solution.routes.append(subRoute)
            solution.routeNum += 1
    return solution


def plot_solution(solution, data):
    df_all = pd.DataFrame({
        'CUST NO.': list(range(len(data.corX))),
        'XCOORD.': data.corX,
        'YCOORD.': data.corY,
        'DEMAND': data.demand,
        'READY TIME': data.readyTime,
        'DUE DATE': data.dueTime,
        'SERVICE TIME': data.serviceTime
    })
    output_file = f'plots/solution_{data.customerNum}.png'
    visualize_vrptw_instance(df_all=df_all, output_file=output_file, routes=solution.routes)


def print_solution(solution, data):
    for index, subRoute in enumerate(solution.routes):
        distance = 0
        load = 0
        for i in range(len(subRoute) - 1):
            distance += data.distanceMatrix[subRoute[i]][subRoute[i + 1]]
            load += data.demand[subRoute[i]]
        print(f"Route-{index + 1} : {subRoute} , distance: {distance} , load: {load}")


def solve(data, M, time_limit=None):
    # 建立模型
    model = Model('VRPTW')
    
    # 模型设置
    model.setParam('MIPGap', 0.05)
    model.setParam('OutputFlag', 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    # Step1: 建立变量索引集合
    X_set = []
    S_set = []
    k_set = [k for k in range(data.vehicleNum)]
    i_set = [i for i in range(data.nodeNum - 1)]  # 不含虚拟终点车场
    j_set = [j for j in range(data.nodeNum)]  # 含虚拟终点车场
    
    for k in k_set:
        for i in i_set:
            for j in j_set:
                if i != j:
                    X_set.append((i, j, k))
    
    for i in range(data.nodeNum):
        for k in range(data.nodeNum):
            S_set.append((i, k))
    
    X_set_tplst = tuplelist(X_set)
    S_set_tplst = tuplelist(S_set)
    
    # Step2: 定义变量
    x = model.addVars(X_set_tplst, vtype=GRB.BINARY, name='x')
    s = model.addVars(S_set_tplst, vtype=GRB.CONTINUOUS, lb=0.0, name='s')
    model.update()
    
    # 定义目标函数
    model.setObjective(
        quicksum(x[i, j, k] * data.distanceMatrix[i][j] for i, j, k in X_set_tplst),
        sense=GRB.MINIMIZE
    )
    
    # 定义约束条件:
    # 1. 客户点服务一次约束
    customer_ids = [i for i in range(1, data.nodeNum - 1)]
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(I, '*', '*')) == 1 for I in customer_ids),
        'customer_once'
    )
    
    # 2. 起点流出约束
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(0, '*', K)) == 1 for K in k_set),
        'start_depot'
    )
    
    # 3. 终点流入约束
    model.addConstrs(
        (quicksum(x[i, j, k] for i, j, k in X_set_tplst.select('*', data.nodeNum - 1, K)) == 1 for K in k_set),
        'end_depot'
    )
    
    # 4. 流平衡约束
    model.addConstrs(
        (quicksum(x[i, h, k] for i, h, k in X_set_tplst.select('*', H, K)) - 
         quicksum(x[h, j, k] for h, j, k in X_set_tplst.select(H, '*', K)) == 0 
         for H, K in product(customer_ids, k_set)),
        'flow_balance'
    )
    
    # 5. 时间窗约束（1）
    model.addConstrs(
        (s[i, k] + data.distanceMatrix[i][j] - M * (1 - x[i, j, k]) <= s[j, k] 
         for i, j, k in X_set_tplst),
        'time_window_constraint'
    )
    
    # 6. 时间窗约束（2）
    model.addConstrs(
        (s[i, k] >= data.readyTime[i] for i, k in S_set_tplst),
        'ready_time'
    )
    model.addConstrs(
        (s[i, k] <= data.dueTime[i] for i, k in S_set_tplst),
        'due_time'
    )
    
    # 7. 容量约束
    model.addConstrs(
        (quicksum(data.demand[i] * x[i, j, k] for i, j, k in X_set_tplst.select('*', '*', K)) <= data.capacity 
         for K in k_set),
        'capacity'
    )
    
    # 求解
    start_time = time.time()
    model.optimize()
    # model.write('VRPTW.lp')
    
    if model.status == GRB.OPTIMAL:
        print("-" * 20, "Solved (Optimal)", '-' * 20)
        print(f"Time consumed: {time.time() - start_time} s")
        print(f"Total Distance: {model.ObjVal}")
        solution = getSolution(data, model)
        print(f"Vehicle Numbers: {solution.routeNum}")
        plot_solution(solution, data)
        print_solution(solution, data)
    elif model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            print("-" * 20, "Solved (Time Limit Reached)", '-' * 20)
            print(f"Time consumed: {time.time() - start_time} s")
            print(f"Total Distance: {model.ObjVal}")
            solution = getSolution(data, model)
            print(f"Vehicle Numbers: {solution.routeNum}")
            plot_solution(solution, data)
            print_solution(solution, data)
        else:
            print("Time limit reached - No feasible solution found")
    else:
        print(f"No Solution found (Status: {model.status})")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='VRPTW Solver')
    parser.add_argument('--data_type', type=str, default='i9', 
                        help='Data type (e.g., i9, c101, r101, rc101)')
    parser.add_argument('--customerNum', type=int, default=None,
                        help='Number of customers to use (default: use all)')
    parser.add_argument('--time', type=float, default=None,
                        help='Time limit in seconds for the solver (default: no limit)')
    args = parser.parse_args()
    
    data_type = args.data_type
    # customerNum = args.customerNum
    customerNum = 10
    time_limit = args.time
    
    if data_type[0] == 'c' or data_type[0] == 'r':
        data_path = f'data/solomon/{data_type}.txt'
    else:
        data_path = f'data/instance/{data_type}.txt'
    
    M = 10000
    data = readData(data_path, customerNum)
    print("-" * 20, "Problem Information", '-' * 20)
    print(f'Data Type: {data_type}')
    print(f'Node Num: {data.nodeNum}')
    print(f'Customer Num: {data.customerNum}')
    print(f'Vehicle Num: {data.vehicleNum}')
    print(f'Vehicle Capacity: {data.capacity}')
    if time_limit is not None:
        print(f'Time Limit: {time_limit} seconds')
    solve(data, M, time_limit)
