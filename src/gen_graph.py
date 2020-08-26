import networkx as nx
import pandas as pd

graph_adj_df = pd.read_csv('ovh_graph.csv')

unique_nodes = np.unique(graph_adj_df['u'])
num_of_nodes = unique_nodes.size

edge_tuples = [tuple(np.array(graph_adj_df)[i]) for i in range(graph_adj_df.shape[0])]
nx_weighted_edge_tuples = map(lambda (x,y,z) : (x,y,{'weight': np.int(z)}), edge_tuples)
nx_unweighted_edge_tuples = map(lambda (x,y,z) : (x,y,{'weight': 1}), edge_tuples)

G_weighted = nx.Graph()
G_weighted.add_nodes_from(unique_nodes)
G_weighted.add_edges_from(nx_weighted_edge_tuples)
G_unweighted = nx.Graph()
G_unweighted.add_nodes_from(unique_nodes)
G_unweighted.add_edges_from(nx_unweighted_edge_tuples)

unique_nodes = np.array(G_unweighted.nodes()) # to get the same ordering of nodes as in the graphs

dist_array = np.array(nx.floyd_warshall_numpy(G_unweighted, nodelist=None, weight='weight')).astype('uint8')
dist_df = pd.DataFrame(dist_array, columns=unique_nodes, index=unique_nodes)

dist_df.to_csv('ovh_dist.csv', index=True, header=True)

#np.savetxt("ovh_dist.csv", dist_mat, delimiter=",")

max_bottleneck_array = np.zeros((num_of_nodes,num_of_nodes))
max_bottleneck = {}
path_bottleneck = {}
for i, source in enumerate(unique_nodes):
    max_bottleneck[source] = {}
    path_bottleneck[source] = {}
    for j, target in enumerate(unique_nodes):
        print source, target, i, j
        if source == target:
            max_bottleneck[source][target] = np.infty
            path_bottleneck[source][target] = [source]
        else:
            temp_max_bottleneck = 0
            for path in nx.all_shortest_paths(G_weighted,source,target):
                curr_bottleneck = min( [ G_weighted[path[k]][path[k+1]]['weight'] for k in range(len(path) - 1) ] )
                if curr_bottleneck > temp_max_bottleneck:
                    temp_max_bottleneck = curr_bottleneck
                    max_bottleneck[source][target] = temp_max_bottleneck
                    path_bottleneck[source][target] = path
        max_bottleneck_array[i,j] = max_bottleneck[source][target]

max_bottleneck_df = pd.DataFrame(max_bottleneck_array, columns=unique_nodes, index=unique_nodes)
max_bottleneck_df.to_csv('ovh_bw.csv', index=True, header=True)
              
            
#for i in range(len(all_pairs['tor']['ams']) - 1):
#    print G[all_pairs['tor']['ams'][i]][all_pairs['tor']['ams'][i+1]]['weight']
