#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cgal_Segmentation_Module as csm
#help(csm)


# In[ ]:


#####2 goals
"""1)  create the final datajoint tables to store the final spine meshs in:
    a. One table to store the segment data (linked to the components table)
    b. One table to store the labeled mesh vertices and faces of the neurons (linked to CleanseMesh)
2) Create the function that goes through and writes the CGAL library for all of the components
"""


# In[ ]:


"""Pseduo code for function that goes through and writes the CGAL library for all of the components

1) Recieve list of neurons to do and a flag that when set will look to
    another table for the clustering parameter for each neuron
2) Pull down the neurons mesh data from cleansedMesh
3) Pull down all the components with the neuron ID that have size > 100
4) For each component:
5) Generate the off file:
    ---------------Way I do it in the blender file----------------
    In load_Neuron_automatic_spine, download whole mesh
    a. Before create the mesh object send faces and verts to filter_verts_and_faces
    b. filter_verts_and_faces:
        downloads the indexes for the compartment
        Only saves off the verts that are mentioned in the indexes
        Only saves off the faces that are mentioned in the indexes
            returns them
    c. builds the off file by:
        Finding the faces that have all indices included in the verts list
        finish with the write_Part_Neuron_Off_file
    -------------------
    1. create the file name string: "neuron_" + str(segment_id) + "_" + str(compartment_type_name) + "_" + str(found_component_index)
    2. get the number of indices and faces
    3. Open them and write them to the file
    4. For the vertices:
        For each index in the vertices blob of the components table, 
         write the coordinates in the index location of the Cleansed mesh table
            while creating a lookup dictionary where it has old_vert_index:new_index  (vert_lookup)
    5. For the faces:
        For each index in the faces blob of the components table,
             Get the new index by (vert_lookup) and save to list
        Write the list to the file
    
    Call the CGAL function to generate the labels
    String calculat the CGAL file name and the CGAL SDF value 
    Write the two lists to the datajoint table (linked to components)
    """




# In[ ]:


import datajoint as dj
import numpy as np
import datetime
import math

#from cloudvolume import CloudVolume
#from collections import Counter
#from funconnect import ta3


# In[ ]:


#setting the address and the username
dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
dj.config['safemode']=True
dj.config["display.limit"] = 20


# user: celiib
# pass: newceliipass
# host: at-database.ad.bcm.edu
# schemas: microns_% and celiib_%


# In[ ]:


schema = dj.schema('microns_ta3p100')
ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')


# In[ ]:


#if temp folder doesn't exist then create it
import os
if (os.path.isdir(os.getcwd() + "/temp")) == False:
    os.mkdir("temp")


# In[ ]:


import os
import pathlib

def generate_component_off_file(neuron_ID, compartment_type, component_id, n_vertex_indices, n_triangle_indices, 
                                vertex_indices, triangle_indices,vertices, triangles):
    
    #get the current file location
    file_loc = pathlib.Path.cwd() / "temp"
    filename = "neuron_" + str(neuron_ID) + "_" + str(compartment_type) + "_" + str(component_id)
    path_and_filename = file_loc / filename
    
    #open the file and start writing to it    
    f = open(str(path_and_filename) + ".off", "w")
    f.write("OFF\n")
    f.write(str(n_vertex_indices) + " " + str(n_triangle_indices) + " 0\n" )
    
    #start writing all of the vertices
    """
        4. For the vertices:
        For each index in the vertices blob of the components table, 
         write the coordinates in the index location of the Cleansed mesh table
            while creating a lookup dictionary where it has old_vert_index:new_index  (vert_lookup)
    """       
    verts_lookup = {}
    for i, vin in enumerate(vertex_indices):
        #get the coordinates of the vertex
        coordinates = vertices[vin]
        #write the coordinates to the off file
        f.write(str(coordinates[0]) + " " + str(coordinates[1]) + " " + str(coordinates[2])+"\n")
        #create lookup dictionary for vertices
        verts_lookup[vin] = i
    
    """    5. For the faces:
        For each index in the faces blob of the components table,
             Get the new index by (vert_lookup) and save to list
        Write the list to the file"""
    for i,fac in enumerate(triangle_indices):
        verts_in_fac = triangles[fac]
        #write the verties to the off file
        f.write("3 " + str(verts_lookup[verts_in_fac[0]]) + " " + str(verts_lookup[verts_in_fac[1]]) + " " + str(verts_lookup[verts_in_fac[2]])+"\n")
        
    
    print("Done making OFF file " + str(filename))
    #return the name of the off file you created and the location
    return str(path_and_filename),str(filename)


# In[ ]:


#################THE ONE WE ARE USING
import cgal_Segmentation_Module as csm
import csv
import decimal
import time
import os

@schema
class ComponentAutoSegment(dj.Computed):
    definition = """
    # creates the labels for the mesh table
    -> ta3p100.Compartment.Component
    clusters     : tinyint unsigned  #what the clustering parameter was set to
    smoothness   : decimal(3,2)             #what the smoothness parameter was set to, number betwee 0 and 1
    ---
    n_triangles  : int unsigned # number of faces
    seg_group    : longblob     # group segmentation ID's for faces from automatic CGAL segmentation
    sdf          : longblob     #  width values for faces from from automatic CGAL segmentation
    median_sdf   : decimal(6,5) # the median width value for the sdf values
    mean_sdf     : decimal(6,5) #the mean width value for the sdf values
    third_q      : decimal(6,5) #the upper quartile for the mean width values
    ninety_perc  : decimal(6,5) #the 90th percentile for the mean width values
    time_updated : timestamp    # the time at which the segmentation was performed
   """
    
    key_source = ta3p100.Compartment.Component & 'n_triangle_indices>100' & [dict(compartment_type=comp) for comp in ['Basal', 'Apical', 'Oblique', 'Dendrite']]
    
    whole_neuron_dicts = dict()
    
    def make(self, key):
        print("key = " + str(key))
        #key passed to function is just dictionary with the following attributes
        """segmentation
        segment_id
        decimation_ratio
        compartment_type
        component_index
        """
        start_time = time.time()
        
        #clusters_default = 18
        smoothness = 0.04

        Apical_Basal_Oblique_default = [12]
        basal_big = [16]

        neuron_ID = key["segment_id"]
        component = (ta3p100.Compartment.Component & key).fetch1()        

        component_id = component["component_index"]
        compartment_type = component["compartment_type"]
        component_size = int(component["n_triangle_indices"])

        print("component_size = " + str(component_size))

        if (compartment_type == "Basal") & (component_size > 160000):
            cluster_list = basal_big
        else:
            cluster_list = Apical_Basal_Oblique_default


        for clusters in cluster_list:
            smoothness = 0.04
            print(str(component["segment_id"]) + " type:" + str(component["compartment_type"]) 
                      + " index:" + str(component["component_index"]) + " cluster:" + str(clusters) 
                  + " smoothness:" + str(smoothness))

            #generate the off file for each component
            #what need to send them:
            """----From cleansed Mesh---
            vertices
            triangles
            ----From component table--
            n_vertex_indices
            n_triangle_indices
            vertex_indices
            triangle_indices"""
            
            if key['segment_id'] not in self.whole_neuron_dicts:
                self.whole_neuron_dicts[key['segment_id']] = (ta3p100.CleansedMesh & 'decimation_ratio=0.35' & dict(segment_id=key['segment_id'])).fetch1()
            
            path_and_filename, off_file_name = generate_component_off_file(neuron_ID, compartment_type, component_id,
                                        component["n_vertex_indices"],
                                        component["n_triangle_indices"],
                                        component["vertex_indices"],
                                        component["triangle_indices"],
                                        self.whole_neuron_dicts[key['segment_id']]["vertices"],
                                        self.whole_neuron_dicts[key['segment_id']]["triangles"])
            
            print(len(component['vertex_indices']), len(component['triangle_indices']))
            
            #will have generated the component file by now so now need to run the segmentation

            csm.cgal_segmentation(path_and_filename,clusters,smoothness)

            #generate the name of the files
            cgal_file_name = path_and_filename + "-cgal_" + str(clusters) + "_"+str(smoothness)
            group_csv_cgal_file = cgal_file_name + ".csv"
            sdf_csv_file_name = cgal_file_name+"_sdf.csv"

            
            try:
                with open(group_csv_cgal_file) as f:
                  reader = csv.reader(f)
                  your_list = list(reader)
                group_list = []
                for item in your_list:
                    group_list.append(int(item[0]))

                with open(sdf_csv_file_name) as f:
                  reader = csv.reader(f)
                  your_list = list(reader)
                sdf_list = []
                for item in your_list:
                    sdf_list.append(float(item[0]))
            except:
                print("no CGAL segmentation for " + str(off_file_name) )
                return

            #print(group_list)
            #print(sdf_list)

            #now write them to the datajoint table  
            #table columns for ComponentAutoSegmentation: segmentation, segment_id, decimation_ratio, compartment_type, component_index, seg_group, sdf
#             print(dict(key,
#                                 clusters=clusters,
#                                 smoothness=smoothness,
#                                 n_triangles=component["n_triangle_indices"],
#                                 seg_group=group_list,
#                                 sdf=sdf_list,
#                                 median_sdf=np.median(sdf_list),
#                                 mean_sdf=np.mean(sdf_list),
#                                 third_q=np.percentile(sdf_list, 75),
#                                 ninety_perc=np.percentile(sdf_list, 90),
#                                 time_updated=str(datetime.datetime.now())[0:19]))
            
            comp_dict = dict(key,
                                clusters=clusters,
                                smoothness=smoothness,
                                n_triangles=component["n_triangle_indices"],
                                seg_group=group_list,
                                sdf=sdf_list,
                                median_sdf=np.median(sdf_list),
                                mean_sdf=np.mean(sdf_list),
                                third_q=np.percentile(sdf_list, 75),
                                ninety_perc=np.percentile(sdf_list, 90),
                                time_updated=str(datetime.datetime.now())[0:19])

            self.insert1(comp_dict)

            #then go and erase all of the files used: the sdf files, 
            real_off_file_name = path_and_filename + ".off"

            files_to_delete = [group_csv_cgal_file,sdf_csv_file_name,real_off_file_name]
            for fl in files_to_delete:
                if os.path.exists(fl):
                    os.remove(fl)
                else:
                    print(fl + " file does not exist")

        print("finished")
        print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


ComponentAutoSegment.populate(reserve_jobs=True)

