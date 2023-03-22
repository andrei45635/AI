def fitness(costs, path):
    fit = 0
    # print(path)
    # print(costs[len(path) - 1])
    # print(len(path))
    for i in range(0, len(path)):
        # print('i', i, 'path1', path[i - 1], 'path2', path[i])
        # print('i', i, 'cost1', costs[path[i - 1] - 1], 'cost2',  costs[path[i] - 1])
        # print('cost', costs[path[i - 1] - 1][path[i] - 1])
        fit += costs[path[i - 1] - 1][path[i] - 1]
    # return 1 / (fit + costs[path[len(path) - 1] - 1][path[0]])
    return fit
