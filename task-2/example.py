from face_embedding import *

# Demonstrate an example of using the module
cluster_pred = find_corresponding_cluster('Celebrity Faces Dataset/Leonardo DiCaprio/005_7fe5b764.jpg')
mapping = sort_into_groups()

for m in mapping:
    if mapping[m][0] == cluster_pred:
        print(f'Embedding probably belongs to {m}')

# Shows which cluster id corresponds to which person along with how well clustering is done
for m in mapping:
    print(f'Cluster {mapping[m][0]} corresponds to {m}, accuracy: {mapping[m][1]}')

