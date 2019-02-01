
import datajoint as dj
import numpy as np
import time


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20

schema = dj.schema('microns_ta3p100')
ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')


@schema
class CleansedMeshOrphan(dj.Computed):
    definition = """
    # Cleansed of floating artifacts and isolated vertices.
    -> ta3p100.DecimationOrphan
    ---
    n_vertices        : bigint
    n_triangles       : bigint
    vertices          : longblob
    triangles         : longblob
    """
    
    def generate_neighborhood(self, triangles, num_vertices):
        neighborhood = dict()
        for i in range(num_vertices):
            neighborhood[i] = set()
        for node1, node2, node3 in triangles:
            neighborhood[node1].update([node2, node3])
            neighborhood[node2].update([node1, node3])
            neighborhood[node3].update([node1, node2])
        return neighborhood
    
    def set_search_first(self, starting_node, neighborhood):
        """
        Modified Depth-First-Search utilizing sets to reduce duplicate checks:

        Neighborhood must be a dict with the keys being the vertex indices!
        """    
        visited_nodes = set()
        temp_stack = set()
        temp_stack.add(starting_node)
        while len(temp_stack) > 0:
            starting_node = temp_stack.pop()
            if starting_node not in visited_nodes:
                visited_nodes.add(starting_node)
                temp_stack.update(neighborhood[starting_node])
        return list(visited_nodes)
    
    def get_connected_portions(self, neighborhood):
        neighborhood_copy = neighborhood.copy()
        portions = []
        while len(neighborhood_copy) > 0:
            starting_node = next(iter(neighborhood_copy))
            portion = self.set_search_first(starting_node, neighborhood_copy)
            for node in portion:
                neighborhood_copy.pop(node)
            portions.append(portion)
        return portions

    def get_largest_portion_index(self, portions):
        portion_lengths = [len(portion) for portion in portions]
        return portion_lengths.index(max(portion_lengths))

    def get_largest_portion(self, portions):
        return portions[self.get_largest_portion_index(portions)]

    def remove_floating_artifacts(self, mesh):    
        mesh_copy = mesh.copy()

        # Generating the neighborhoods gets quite expensive for full resolution meshes, but the searches are extremely quick.
        neighborhood = self.generate_neighborhood(mesh_copy['triangles'], len(mesh_copy['vertices']))
        portions = self.get_connected_portions(neighborhood)

        main_mesh_body_index = self.get_largest_portion_index(portions)
        triangle_removal_nodes = portions[main_mesh_body_index:] + portions[:main_mesh_body_index + 1]

        new_triangles = []
        main_body_portion = set(self.get_largest_portion(portions))
        for i, triangle in enumerate(mesh_copy['triangles']):
            node1 = triangle[0]
            if node1 in main_body_portion:
                new_triangles.append(triangle)
        mesh_copy['triangles'] = np.array(new_triangles)

        return mesh_copy

    def remove_isolated_vertices(self, mesh):
        mesh_copy = mesh.copy()

        neighborhood = self.generate_neighborhood(mesh_copy['triangles'], len(mesh_copy['vertices']))
        isolated_nodes = [portion.pop() for portion in self.get_connected_portions(neighborhood) if len(portion) == 1]

        vertices = mesh_copy['vertices']
        triangles = mesh_copy['triangles']

        if len(isolated_nodes) > 0:
            num_isolated_nodes_passed = 0
            isolated_nodes_set = set(isolated_nodes)
            count_to_decrement = np.zeros(len(vertices))
            for i in range(len(vertices)):
                if i in isolated_nodes_set:
                    num_isolated_nodes_passed += 1
                else:
                    count_to_decrement[i] = num_isolated_nodes_passed

            for i, triangle in enumerate(triangles):
                start = time.time()
                node1, node2, node3 = triangle
                triangles[i][0] -= count_to_decrement[node1]
                triangles[i][1] -= count_to_decrement[node2]
                triangles[i][2] -= count_to_decrement[node3]

            vertex_list = list(vertices)
            for i, isolated_node in enumerate(isolated_nodes):
                vertex_list.pop(isolated_node - i)

        mesh_copy['vertices'] = np.array(vertex_list)

        return mesh_copy
    
    key_source = ta3p100.DecimationOrphan
    
    def make(self, key):
        full_start = time.time()
        
        print(key['segment_id'], key['decimation_ratio'], ":")
        start = time.time()
                
        mesh = (ta3p100.DecimationOrphan & key).fetch1()
        print(key['segment_id'], "mesh fetched.", time.time() - start)
        start = time.time()
                
        neighborhood = self.generate_neighborhood(mesh['triangles'], len(mesh['vertices']))
        print(key['segment_id'] , "neighborhood generated.", time.time() - start)
        start = time.time()
        
        mesh = self.remove_floating_artifacts(mesh)
        print(key['segment_id'], "floating artifacts removed.", time.time() - start)
        start = time.time()
        
        mesh = self.remove_isolated_vertices(mesh)
        print(key['segment_id'], "isolated nodes removed.", time.time() - start)
        start = time.time()
                
        key['n_vertices'] = len(mesh['vertices'])
        key['n_triangles'] = len(mesh['triangles'])
        key['vertices'] = mesh['vertices']
        key['triangles'] = mesh['triangles']
        
        self.insert1(key)
        print(key['segment_id'], "key successfully inserted.", time.time() - start)
        start = time.time()
        
        print("This took ", time.time() - full_start, "seconds.")
        print()


# In[8]:


start = time.time()
CleansedMeshOrphan().populate()#reserve_jobs=True)
print("Final:", time.time() - start)





