# Samuel Scalise
# Undergraduate
# Python
# Project 3 Machine Learning
# K-means Clustering
# Need Python standard add-on csv(comes with every installation of Python >= 2.7)

# include csv (did not end up using, but kept in incase of further development of the program)
#   CSV is just the standard comma separated format for text files

# Used package matplotlib from (https://matplotlib.org):
#       Used Only to create the scatter plot graph of the randomized points
#       I figured it would be okay, because the project allowed even hand drawn graphs
#       Allows for ease in creating a scatter plot graph with different colored points for individual clusters
#       !Did not use in implementation of k-means!
#       numpy package is included with matplotlib install
#           Used to allow for object and list implementation in matplotlib
#           only used for implementation of matplotlib
#       matplotlib is a simple pip install:
#           windows: python -m pip install -U matplotlib (for matplotlib install)
#           windows: python -m pip install -U pip (If pip is not installed or out of date, run first)

# Summary:
#       This program runs j-means to cluster random plot data using Euclidean distance and the Manhattan method.
#       Project is defined as having two dimensions:
#           X: 1 <= x < inf
#           Y: -inf < y <= 100
#       Project also defines that 20 plots are needed.
#       Project also defines that both k = 2 and k = 4 are needed to be found

import math
#   import random is needed to choose random numbers
#   import math is needed to perform math functions
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
#   define a list for the features( [x, y] in projects case)
#   define the number of plots
#   define max x and max y
#   define min x and min y
#   define the k as a list
#   define the k values in the list
#   sse_clusters is a list for the clusters after rounds of sse
#   manhattan_clusters is a list for the clusters after rounds of Manhattan distance
#   k_count is the amount of k's given (two in this case)
#   sse_all is the list containing the clusters for each Euclidean k-means clustering for each k
#   manhattan_all is the list containing the clusters for each k-means manhattan distances for each k
#   set the seed for random (This is so the project results can be redone/remade to ensure the projects output validity)
#       Just picked seed = 100 randomly
#       Could improve by making it user defined or a random number based on the system clock if no seed given
#   Graph dimensions is a hard coded max size for the graph plots
#       This is used to make the range and domain of random plots
#       Ignores the max and min values of plots
#       Used because max and min for plot points may not be given and can't have ranges/domains going to infinity
#   All variables are hard coded because they are given in the project description
#       making them variable and/or user defined is a small step from this point in case of future program development
features = []
feature_count = 2
plot_number = 20
feature_max = [None, 100]
feature_min = [1.0, None]
k = []
k_count = 2
k.append(2)
k.append(4)
seed = 100
graph_dimensions = [-1000, 1000, -1000, 1000]
sse_clusters = []
manhattan_clusters = []
sse_all = []
manhattan_all = []
maxs = []
mins = []

#   Now set the features list up based on feature count for use later
#   Set see_all for later
#   set manhattan_all for later
#   This section would be where more set up would be done if designing code for more than just the project
#   define the function for generating random points
for count in range(feature_count):
    features.append([])
for count in range(k_count):
    sse_all.append([])
    manhattan_all.append([])


def random_plot(sed):
    random.seed(sed)
    for feat in range(feature_count):
        #   create for loop in range 0-plot_number
        #   This loop is for x's
        for pos in range(plot_number):
            #   Set a flag for loop to ensure random number is within the range
            invalid_flg = True
            while invalid_flg:
                invalid_flg = False
                temp = random.randint(graph_dimensions[0], graph_dimensions[1]) * (random.randint(1, 100) / 100)
                if feature_min[feat]:
                    if temp < feature_min[feat]:
                        invalid_flg = True
                if feature_max[feat]:
                    if temp > feature_max[feat]:
                        invalid_flg = True
                if not invalid_flg:
                    features[feat].append(temp)


#   Now we define a function for a single round of k-means clustering
#   This program goes from bottom up, or all points are clusters to a single cluster
#   This function bases these clusters off of SSE and Manhattan distance
def k_means():
    k_spot = 0
    for centers in k:
        for pos in range(plot_number):
            temp = []
            for feat in range(feature_count):
                temp.append(features[feat][pos])
            sse_clusters.append([temp])

        for pos in range(plot_number):
            temp = []
            for feat in range(feature_count):
                temp.append(features[feat][pos])
            manhattan_clusters.append([temp])

        cluster_total = plot_number
        while not cluster_total == centers:
            sse_min = None
            man_min = None
            first_pos = 0
            second_pos = 1
            first_pos_man = 0
            second_pos_man = 1
            for first_cluster in range(cluster_total - 1):
                sse = 0
                manhattan = 0
                second_cluster = first_cluster + 1
                for first_point in sse_clusters[first_cluster]:
                    for second_point in sse_clusters[second_cluster]:
                        sse += math.sqrt((second_point[0] - first_point[0])**2 + (second_point[1] - first_point[1])**2)
                        x = second_point[0] - first_point[0]
                        if x < 0:
                            x = x * -1
                        y = second_point[0] - first_point[1]
                        if y < 0:
                            y = y * 1
                        manhattan += (x + y)
                if (not sse_min) or (sse < sse_min):
                    sse_min = sse
                    first_pos = first_cluster
                    second_pos = second_cluster
                if (not man_min) or (manhattan < man_min):
                    man_min = manhattan
                    first_pos_man = first_cluster
                    second_pos_man = second_cluster
            for item in sse_clusters[second_pos]:
                sse_clusters[first_pos].append(item)
            sse_clusters.pop(second_pos)
            for item in manhattan_clusters[second_pos_man]:
                manhattan_clusters[first_pos_man].append(item)
            manhattan_clusters.pop(second_pos_man)
            cluster_total -= 1
        for cluster in sse_clusters:
            sse_all[k_spot].append(cluster)
        for cluster in manhattan_clusters:
            manhattan_all[k_spot].append(cluster)
        k_spot += 1
        sse_clusters.clear()
        manhattan_clusters.clear()


#   Now we need to define a function to find the max, min, total intra-cluster distance, and individual cluster distance
def individual_cluster_intradistance(cluster):
    p_count = 0
    distance = 0
    for p in cluster:
        p_count += 1
    for position in range(p_count):
        if not position == (p_count - 1):
            distance += math.sqrt((cluster[position + 1][0] - cluster[position][0])**2 + (cluster[position + 1][1] - cluster[position][1])**2)
    return distance

def maxfinder(clusters):
    cluster_count = 0
    maxs.clear()
    mins.clear()
    for cluster in clusters:
        cluster_count += 1
    for spot in range(cluster_count):
        maxs.append([])
        mins.append([])
    for cluster in range(cluster_count):
        max_pos1 = 0
        max_pos2 = 1
        min_pos1 = 0
        min_pos2 = 1
        mind = None
        maxd = None
        for point in clusters[cluster]:
            # inner loop
            for cluster_two in range(cluster_count):
                if not cluster == cluster_two:
                    for point_two in clusters[cluster_two]:
                        d = math.sqrt((point_two[0]-point[0])**2 + (point_two[1]-point[1])**2)
                        if (not mind) or (d < mind):
                            mind = d
                            min_pos1 = cluster
                            min_pos2 = cluster_two
                        if (not maxd) or (d > maxd):
                            maxd = d
                            max_pos1 = cluster
                            max_pos2 = cluster_two
        maxs[cluster].append(max_pos1)
        maxs[cluster].append(max_pos2)
        maxs[cluster].append(maxd)
        mins[cluster].append(min_pos1)
        mins[cluster].append(min_pos2)
        mins[cluster].append(mind)




random_plot(seed)
k_means()
k_pos = 0
intra_distance_sse = []
intra_distance_man = []
total_intra_sse = []
total_intra_man = []

for ks in range(k_count):
    intra_distance_sse.append([])
for element in sse_all:
    for item in element:
        temp = individual_cluster_intradistance(item)
        intra_distance_sse[k_pos].append(temp)
    k_pos += 1

k_pos = 0
for ks in range(k_count):
    intra_distance_man.append([])
for element in manhattan_all:
    for item in element:
        temp = individual_cluster_intradistance(item)
        intra_distance_man[k_pos].append(temp)
    k_pos += 1

for sse_distance in intra_distance_sse:
    total = 0
    for sd in sse_distance:
        total += sd
    total_intra_sse.append(total)

for man_distance in intra_distance_man:
    total = 0
    for mn in man_distance:
        total += mn
    total_intra_man.append(total)

with open('sseresult.txt', 'w') as f:
    k_pos = 0
    element_pos = 0
    for element in sse_all:
        maxfinder(element)
        f.write(f"k = {k[k_pos]}:\n----------")
        for position in range(k[k_pos]):
            f.write(f"\ncluster {position}:\n")
            f.write("\tPoints:\n")
            for point in element[position]:
                f.write(f"\t\t{point}\n")
            f.write(f"\tMax:\n")
            f.write(f"\t\t Cluster pair = (cluster {maxs[position][0]}, cluster {maxs[position][1]})\n")
            f.write(f"\t\t Cluster pair distance = {maxs[position][2]}\n")
            f.write(f"\tMin:\n")
            f.write(f"\t\t Cluster pair = (cluster {mins[position][0]}, cluster {mins[position][1]})\n")
            f.write(f"\t\t Cluster pair distance = {mins[position][2]}\n")
            f.write(f"\tIntracluster distance:\n")
            f.write(f"\t\t{intra_distance_sse[k_pos][position]}\n")
        f.write(f"\nTotal Distance between points = {total_intra_sse[k_pos]}\n\n")
        k_pos += 1

with open('manresult.txt', 'w') as f:
    k_pos = 0
    element_pos = 0
    for element in manhattan_all:
        maxfinder(element)
        f.write(f"k = {k[k_pos]}:\n----------")
        for position in range(k[k_pos]):
            f.write(f"\ncluster {position}:\n")
            f.write("\tPoints:\n")
            for point in element[position]:
                f.write(f"\t\t{point}\n")
            f.write(f"\tMax:\n")
            f.write(f"\t\t Cluster pair = (cluster {maxs[position][0]}, cluster {maxs[position][1]})\n")
            f.write(f"\t\t Cluster pair distance = {maxs[position][2]}\n")
            f.write(f"\tMin:\n")
            f.write(f"\t\t Cluster pair = (cluster {mins[position][0]}, cluster {mins[position][1]})\n")
            f.write(f"\t\t Cluster pair distance = {mins[position][2]}\n")
            f.write(f"\tIntracluster distance:\n")
            f.write(f"\t\t{intra_distance_man[k_pos][position]}\n")
        f.write(f"\nTotal Distance between points = {total_intra_man[k_pos]}\n\n")
        k_pos += 1

cx = []
cy = []
for k_pos in range(k_count):
    mark = 2
    for cluster in sse_all[k_pos]:
        for pt in cluster:
            cx.append(pt[0])
            cy.append(pt[1])
        mark += 1
        markr = (mark, 0, 90)
        plt.scatter(cx, cy, s=None, c=None, marker=markr)
        cx.clear()
        cy.clear()
    fig = "euclidean" + str(k_pos) + ".png"
    plt.savefig(fig)
    mark = 2
    for cluster in manhattan_all[k_pos]:
        for pt in cluster:
            cx.append(pt[0])
            cy.append(pt[1])
        mark += 1
        markr = (mark, 0, 90)
        plt.scatter(cx, cy, s=None, c=None, marker=markr)
        cx.clear()
        cy.clear()
    fig = "manhatten" + str(k_pos) + ".png"
    plt.savefig(fig)
    mark = 2
