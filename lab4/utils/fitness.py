def fitness(costs, path):
    fit = 0
    for i in range(0, len(path)):
        fit += costs[path[i - 1] - 1][path[i] - 1]
    return fit
