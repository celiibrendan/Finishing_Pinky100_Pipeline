#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datajoint as dj
import numpy as np
import time


# In[2]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_ta3p100')
ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
ta3 = dj.create_virtual_module('ta3', 'microns_ta3')


# In[3]:


ta3p100.CurrentSegmentation()


# In[4]:


pyramidal_cell_rel = ta3p100.AllenSoma & (ta3p100.AllenSomaClass & 'cell_class="excitatory"')


# In[ ]:


true_start = time.time()


# In[ ]:


ta3p100.Decimation & ta3p100.CurrentSegmentation & 'decimation_ratio=0.10' & pyramidal_cell_rel


# In[ ]:





# In[ ]:


# @schema
# class CleansedMesh(dj.Computed):
#     definition = """
#     # Cleansed of floating artifacts and isolated vertices.
#     -> ta3p100.Decimation
#     ---
#     n_vertices        : bigint
#     n_triangles       : bigint
#     vertices          : longblob
#     triangles         : longblob
#     """
    
#     def generate_neighborhood(self, triangles, num_vertices):
#         neighborhood = dict()
#         for i in range(num_vertices):
#             neighborhood[i] = set()
#         for node1, node2, node3 in triangles:
#             neighborhood[node1].update([node2, node3])
#             neighborhood[node2].update([node1, node3])
#             neighborhood[node3].update([node1, node2])
#         return neighborhood
    
#     def set_search_first(self, starting_node, neighborhood):
#         """
#         Modified Depth-First-Search utilizing sets to reduce duplicate checks:

#         Neighborhood must be a dict with the keys being the vertex indices!
#         """    
#         visited_nodes = set()
#         temp_stack = set()
#         temp_stack.add(starting_node)
#         while len(temp_stack) > 0:
#             starting_node = temp_stack.pop()
#             if starting_node not in visited_nodes:
#                 visited_nodes.add(starting_node)
#                 temp_stack.update(neighborhood[starting_node])
#         return list(visited_nodes)
    
#     def get_connected_portions(self, neighborhood):
#         neighborhood_copy = neighborhood.copy()
#         portions = []
#         while len(neighborhood_copy) > 0:
#             starting_node = next(iter(neighborhood_copy))
#             portion = self.set_search_first(starting_node, neighborhood_copy)
#             for node in portion:
#                 neighborhood_copy.pop(node)
#             portions.append(portion)
#         return portions

#     def get_largest_portion_index(self, portions):
#         portion_lengths = [len(portion) for portion in portions]
#         return portion_lengths.index(max(portion_lengths))

#     def get_largest_portion(self, portions):
#         return portions[self.get_largest_portion_index(portions)]

#     def remove_floating_artifacts(self, mesh):    
#         mesh_copy = mesh.copy()

#         # Generating the neighborhoods gets quite expensive for full resolution meshes, but the searches are extremely quick.
#         neighborhood = self.generate_neighborhood(mesh_copy['triangles'], len(mesh_copy['vertices']))
#         portions = self.get_connected_portions(neighborhood)

#         main_mesh_body_index = self.get_largest_portion_index(portions)
#         triangle_removal_nodes = portions[main_mesh_body_index:] + portions[:main_mesh_body_index + 1]

#         new_triangles = []
#         main_body_portion = set(self.get_largest_portion(portions))
#         for i, triangle in enumerate(mesh_copy['triangles']):
#             node1 = triangle[0]
#             if node1 in main_body_portion:
#                 new_triangles.append(triangle)
#         mesh_copy['triangles'] = np.array(new_triangles)

#         return mesh_copy

#     def remove_isolated_vertices(self, mesh):
#         mesh_copy = mesh.copy()

#         neighborhood = self.generate_neighborhood(mesh_copy['triangles'], len(mesh_copy['vertices']))
#         isolated_nodes = [portion.pop() for portion in self.get_connected_portions(neighborhood) if len(portion) == 1]

#         vertices = mesh_copy['vertices']
#         triangles = mesh_copy['triangles']

#         if len(isolated_nodes) > 0:
#             num_isolated_nodes_passed = 0
#             isolated_nodes_set = set(isolated_nodes)
#             count_to_decrement = np.zeros(len(vertices))
#             for i in range(len(vertices)):
#                 if i in isolated_nodes_set:
#                     num_isolated_nodes_passed += 1
#                 else:
#                     count_to_decrement[i] = num_isolated_nodes_passed

#             for i, triangle in enumerate(triangles):
#                 start = time.time()
#                 node1, node2, node3 = triangle
#                 triangles[i][0] -= count_to_decrement[node1]
#                 triangles[i][1] -= count_to_decrement[node2]
#                 triangles[i][2] -= count_to_decrement[node3]

#             vertex_list = list(vertices)
#             for i, isolated_node in enumerate(isolated_nodes):
#                 vertex_list.pop(isolated_node - i)

#         mesh_copy['vertices'] = np.array(vertex_list)

#         return mesh_copy
    
#     key_source = ta3p100.Decimation & ta3p100p100.CurrentSegmentation & 'decimation_ratio=0.35' & pyramidal_cell_rel 
    
#     def make(self, key):
#         full_start = time.time()
        
#         print(key['segment_id'], key['decimation_ratio'], ":")
#         start = time.time()
                
#         mesh = (ta3p100.Decimation & key).fetch1()
#         print(key['segment_id'], "mesh fetched.", time.time() - start)
#         start = time.time()
                
#         neighborhood = self.generate_neighborhood(mesh['triangles'], len(mesh['vertices']))
#         print(key['segment_id'] , "neighborhood generated.", time.time() - start)
#         start = time.time()
        
#         mesh = self.remove_floating_artifacts(mesh)
#         print(key['segment_id'], "floating artifacts removed.", time.time() - start)
#         start = time.time()
        
#         mesh = self.remove_isolated_vertices(mesh)
#         print(key['segment_id'], "isolated nodes removed.", time.time() - start)
#         start = time.time()
                
#         key['n_vertices'] = len(mesh['vertices'])
#         key['n_triangles'] = len(mesh['triangles'])
#         key['vertices'] = mesh['vertices']
#         key['triangles'] = mesh['triangles']
        
#         self.insert1(key)
#         print(key['segment_id'], "key successfully inserted.", time.time() - start)
#         start = time.time()
        
#         print("This took ", time.time() - full_start, "seconds.")
#         print()


# In[ ]:


# start = time.time()
# CleansedMesh().populate()#reserve_jobs=True)
# print("Final:", time.time() - start)


# In[ ]:





# In[ ]:


#############################################################################################################

def generate_neighborhood(triangles, num_vertices):
    neighborhood = dict()
    for i in range(num_vertices):
        neighborhood[i] = set()
    for node1, node2, node3 in triangles:
        neighborhood[node1].update([node2, node3])
        neighborhood[node2].update([node1, node3])
        neighborhood[node3].update([node1, node2])
    return neighborhood

def compress_compartments(neighborhood, vertex_labels):
    boundary_clusters = dict()
    for unique_label in np.unique(vertex_labels):
        boundary_clusters[unique_label] = dict()#list()

    starting_node = 0 # This assumes that there are no disconnected portions... I should actually figure out exactly what's going on here.
    visited_nodes = set()
    temp_stack = set()
    temp_stack.add(starting_node)    
    while len(temp_stack) > 0:
        starting_node = temp_stack.pop()
        if starting_node not in visited_nodes:
            same_label_neighbors = set()
            node_label = vertex_labels[starting_node]
            is_on_boundary = False
            for neighboring_node in neighborhood[starting_node]: # Think about if I truly need the same labeled neighbors...
                                                                 # Only way for it to be truly self contained right?
                if node_label == vertex_labels[neighboring_node]:
                    same_label_neighbors.add(neighboring_node)
                else:
                    is_on_boundary = True
            if is_on_boundary:
#                 boundary_clusters[node_label].append((starting_node, same_label_neighbors))
                boundary_clusters[node_label][starting_node] = same_label_neighbors
                
            visited_nodes.add(starting_node)
            temp_stack.update(neighborhood[starting_node])
    return boundary_clusters

def _separate_compartment(neighborhood, cluster, boundary_points):
    components = dict()
    compartment_index = 0
    while len(cluster) > 0:
        visited_nodes = set()
        temp_stack = set()
        temp_stack.add(next(iter(cluster)))
        boundaries_hit = set()
        while len(temp_stack) > 0:
            starting_node = temp_stack.pop()
            if starting_node not in visited_nodes:
                visited_nodes.add(starting_node)
                if starting_node in boundary_points:
                    boundaries_hit.add(starting_node)
                    temp_stack.update(cluster[starting_node])
                else:
                    temp_stack.update(neighborhood[starting_node])
        [cluster.pop(boundary_hit) for boundary_hit in boundaries_hit]        
        components[compartment_index] = visited_nodes
        compartment_index += 1
    return components

def separate_compartments(neighborhood, boundary_clusters):
    compartment_components = dict()
    boundary_clusters_copy = boundary_clusters.copy()
    for label, boundary_cluster in boundary_clusters_copy.items():
        cluster = dict()
        boundary_points = set()
        for node, neighbors in boundary_cluster.items():
            boundary_points.add(node)
            cluster[node] = neighbors
        components = _separate_compartment(neighborhood, cluster, boundary_points)
        compartment_components[label] = components
    return compartment_components
        
############################################################################################################# For Below

@schema
class Compartment(dj.Computed):
    definition = """
    -> ta3p100.CleansedMesh
    ---
    """

    class Component(dj.Part):
        definition = """
        -> Compartment
        compartment_type   : varchar(16)        # Basal, Apical, spine head, etc.
        component_index    : smallint unsigned  # Which sub-compartment of a certain label this is.
        ---
        n_vertex_indices   : bigint
        n_triangle_indices : bigint
        vertex_indices     : longblob           # preserved indices of each vertex of this sub-compartment
        triangle_indices   : longblob           # preserved indices of each triangle of this sub-compartment
        """
    
    key_source = ta3p100.CleansedMesh & ta3p100.CurrentSegmentation & 'decimation_ratio=0.35' & ta3p100.CoarseLabel.proj()

    def make(self, key):
        def generate_triangle_neighborhood(triangles):
            """
            Maps each vertex node to every triangle they appear in.
            """
            triangle_neighborhood = dict()
            for i in range(len(triangles)):
                triangle_neighborhood[i] = set()
            for i, (node1, node2, node3) in enumerate(triangles):
                triangle_neighborhood[node1].add(i)
                triangle_neighborhood[node2].add(i)
                triangle_neighborhood[node3].add(i)
            return triangle_neighborhood
        
        def generate_component_keys(key, components, triangles, triangle_neighborhood, labeled_triangles):
            for label_key, compartment in components.items():
                for component_index, component in compartment.items():
                    try:
                        label_name = (ta3.LabelKey & dict(numeric=label_key)).fetch1('description')
                    except:
                        label_name = str(label_key)
                        
                    vertex_indices = np.array(list(component))
                    triangle_indices = np.unique([triangle_index for node in component
                                                  for triangle_index in triangle_neighborhood[node]
                                                  if labeled_triangles[triangle_index] == label_key])
                    set_vertex_indices = set(vertex_indices)
                    true_triangle_indices = []
                    for triangle_index in triangle_indices:
                        node1, node2, node3 = triangles[triangle_index]
                        if node1 in set_vertex_indices:
                            if node2 in set_vertex_indices:
                                if node3 in set_vertex_indices:
                                    true_triangle_indices.append(triangle_index)                        
                    triangle_indices = np.array(true_triangle_indices)
                    yield dict(key,
                               compartment_type=label_name,
                               component_index=component_index,
                               n_vertex_indices=len(vertex_indices),
                               n_triangle_indices=len(triangle_indices),
                               vertex_indices=vertex_indices,
                               triangle_indices=triangle_indices)
        
        start = time.time()
        
        mesh = (ta3p100.CleansedMesh & key).fetch1()
        labels = (ta3p100.CoarseLabel & key).fetch1()
        
        neighborhood = generate_neighborhood(mesh['triangles'], len(mesh['vertices']))
        boundary_clusters = compress_compartments(neighborhood, labels['vertices'])
        components = separate_compartments(neighborhood, boundary_clusters)
        triangle_neighborhood = generate_triangle_neighborhood(mesh['triangles'])

        self.insert1(key)
        Compartment.Component().insert(generate_component_keys(key, components, mesh['triangles'],
                                                               triangle_neighborhood, labels['triangles']))

        print(key['segment_id'], "finished separating components:", time.time() - start)


# In[ ]:


Compartment.populate(reserve_jobs=True)


# In[ ]:


print(time.time() - true_start)


# In[ ]:


#(schema.jobs & "table_name='__compartment'").delete()


# In[ ]:


ta3.LabelKey()


# In[ ]:




