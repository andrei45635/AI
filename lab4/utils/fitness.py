def fitness(costs, path):
    fit = 0
    for i in range(0, len(path) - 1):
        fit += costs[path[i]][path[i + 1]]
    return fit + costs[path[len(path) - 1]][path[0]]
