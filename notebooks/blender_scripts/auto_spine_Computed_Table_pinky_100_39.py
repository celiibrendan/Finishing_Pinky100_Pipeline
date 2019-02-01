import bpy

#This one will pull down some of the larger segments from the datajoint 
#table and then apply the automatic segmentation to them

#######Steps##############
'''1) Get the neuron the person wants to look at
2) Import the neuron and generate edges
3) Get the compartment_type person wants
4) Find the component_index that corresponds to the biggest one because that is the one we want
5) Delete all the edges, faces and vertices that do not correspond to these labels
6) Generate an OFF file for the current segment
7) Run the OFF file through the CGAL segmentation algorithm using the INPUT PARAMETERS
8) Run the auto spine labeler using the CGAL segmentation list
9) Label the colors of the auto labeled spines and show the final product
10) Output stats to a csv so they can be analyzed'''

####How to import from the segment table

import datajoint as dj
import numpy as np
import datetime
import math
from mathutils import Vector

dj.config['database.host'] = '10.28.0.34'
dj.config['database.user'] = 'celiib'
dj.config['database.password'] = 'newceliipass'
#will state whether words are shown or not
dj.config['safemode']=True
print(dj.conn(reset=True))

def select_Neuron(ob_name):
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    bpy.context.scene.objects.active = None

    # selection
    obj = bpy.data.objects[ob_name]
    bpy.context.scene.objects.active = obj
    """for obj in bpy.data.objects:
        if "neuron" in obj.name:
            obj.select = True
            bpy.context.scene.objects.active = obj
            print("object was found and active")
            break"""

#1) Get the neuron the person wants to look at
#2) Import the neuron and generate edges

def filter_verts_and_faces(key,verts,faces):
    #go and get the triangles and the vertices from the database    
    """compartment_type
    decimation_ratio
    segmentation
    segment_id"""
    
    component_key = dict(segmentation=key["segmentation"],
                    segment_id=key["segment_id"],
                    decimation_ratio=float(key["decimation_ratio"]),
                    compartment_type=key["compartment_type"],
                    component_index=key["component_index"])
                    
    
    verts_label, triangles_label = (ta3p100.Compartment.Component & component_key).fetch('vertex_indices','triangle_indices')
    
    verts_label = verts_label.tolist()[0]
    triangles_label = triangles_label.tolist()[0]
    
    verts_keep = []
    faces_keep = []
    verts_lookup = {}
    
    for i,ver in enumerate(verts_label):
        verts_keep.append(verts[ver])
        verts_lookup[ver] = i
    
    #generate the new face labels
    for fac in triangles_label:
        faces_with_verts = faces[fac]
        new_tuple = []
        for v in faces_with_verts:
            new_tuple.append(verts_lookup[v])
        
        faces_keep.append(new_tuple)
    #check that the new verts and faces to return are same length as the indices
    """if len(triangles_label) != len(faces_keep) or len(verts_label) != len(verts_keep):
        print("ERROR THE FILTERED LABELS ARE NOT THE SAME SIZE AS THE INDICES LISTS!")"""
     
    return verts_keep,faces_keep

whole_neuron_dicts = dict()
def load_Neuron_automatic_spine(key):
    ID = key['segment_id']
    compartment_type = key['compartment_type']
    compartment_index = key['component_index']
    print("inside load Neuron")
    
    #neuron_data = ((mesh_Table & "segment_ID="+ID).fetch(as_dict=True))[0]
    if ID not in whole_neuron_dicts:
        whole_neuron_dicts[ID] = (ta3p100.CleansedMesh & 'decimation_ratio=0.35' & dict(segment_id=ID)).fetch1()
    
    verts = whole_neuron_dicts[ID]['vertices'].astype(dtype=np.uint32).tolist()
    faces = whole_neuron_dicts[ID]['triangles'].astype(dtype=np.uint32).tolist()
    
    #could filter the verts and the faces here for just the ones we want
    verts,faces = filter_verts_and_faces(key,verts,faces)
    
    mymesh = bpy.data.meshes.new("neuron-"+str(ID))
    mymesh.from_pydata(verts, [], faces)
 
    mymesh.update(calc_edges=True)
    mymesh.calc_normals()

    object = bpy.data.objects.new("neuron-"+str(ID), mymesh)
    #object.location = bpy.context.scene.cursor_location
    object.location = Vector((0,0,0))
    bpy.context.scene.objects.link(object)
    
    object.lock_location[0] = True
    object.lock_location[1] = True
    object.lock_location[2] = True
    object.lock_scale[0] = True
    object.lock_scale[1] = True
    object.lock_scale[2] = True

    object.rotation_euler[0] = 1.5708
    object.rotation_euler[1] = 0
    object.rotation_euler[2] = 0

    object.lock_rotation[0] = True
    object.lock_rotation[1] = True
    object.lock_rotation[2] = True


    #set view back to normal:
    #set_View()

    #run the setup color command
    #bpy.ops.object.select_all(action='TOGGLE')
    
    #create_local_colors(object)

    #make sure in solid mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
    
    return object.name




##write the OFF file for the neuron

def write_Part_Neuron_Off_file(verts_for_off,faces_for_off,faces_indexes_for_off,segment_id,compartment_type_name,found_component_index,file_loc):
    print('inside write_Part_neuron')
    num_vertices = (len(verts_for_off))
    num_faces = len(faces_indexes_for_off)
    
    file_location = file_loc
    filename = "neuron_" + str(segment_id) + "_" + str(compartment_type_name) + "_" + str(found_component_index)
    f = open(file_location + filename + ".off", "w")
    f.write("OFF\n")
    f.write(str(num_vertices) + " " + str(num_faces) + " 0\n" )
    
    ob = bpy.context.object
    verts_raw = ob.data.vertices
    
    #iterate through and write all of the vertices in the file
    verts_lookup = {}
    
    counter = 0
    for vert_num in verts_for_off:
        f.write(str(verts_raw[vert_num].co[0]) + " " + str(verts_raw[vert_num].co[1]) + " " + str(verts_raw[vert_num].co[2])+"\n")
        verts_lookup[vert_num] = counter
        
        counter += 1
        
    faces_lookup_reverse = []
    counter = 0
    
    print("finished writing verts")
    for i in range(0,len(faces_indexes_for_off)):
        face_indices = faces_indexes_for_off[i]
        f.write("3 " + str(verts_lookup[face_indices[0]]) + " " + str(verts_lookup[face_indices[1]]) + " " + str(verts_lookup[face_indices[2]])+"\n")
        faces_lookup_reverse.append(faces_for_off[i])
        counter += 1
  
    print("finished writing faces")
    print("done_writing_off_file")
    #f.write("end")
    return filename,faces_lookup_reverse


import random

def get_cgal_data_and_label(key,ob_name):
       
    #store the group_segmentation in the traingle labels from datajoint        
    component_data = (ta3p100.ComponentAutoSegment() & key).fetch(as_dict=True)
    if component_data == []:
        return [], []
    else:
        component_data = component_data[0]
        
    triangles_labels = component_data["seg_group"].tolist()
    #activate the current object
    select_Neuron(ob_name)
    ob = bpy.context.object
    
    
    me = ob.data
    
    #print("starting to hide everything")
    #iterate through all of the vertices
    verts_raw = ob.data.vertices
    #print(len(active_verts_raw))
    
    edges_raw = ob.data.edges
    
    #print(len(active_edges_raw))
    
    faces_raw = ob.data.polygons
    
    #gets a list of the unique labels
    unique_segments = list(Counter(triangles_labels).keys())
    
    
    segmentation_length = len(unique_segments) # equals to list(set(words))
    #print(segmentation_length)

    #makes a dictionary that maps the unique segments to a number from range(0,len(unique_seg))
    unique_index_dict = {unique_segments[x]:x for x in range(0,segmentation_length)}
    
    
    #print("unique_index_dict = " + str(len(unique_index_dict)))
    #print("triangle_labels = " + str(len(triangles_labels)))
    #adds all of the labels to the faces
    max_length = len(triangles_labels)
    
    #just iterate and add them to the faces
    #here is where need to get stats for sdf numbers
    
    
    labels_list = []
    for tri in triangles_labels:

        #assembles the label list that represents all of the faces
        labels_list.append(str(unique_index_dict[tri])) 
    
    select_Neuron(ob_name)
    
    
    #make sure in solid mode
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'SOLID' # set the viewport shading to rendered
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    #these variables are set in order to keep the functions the same as FINAL_importing_auto_seg.py
    newname = ob.name
    print("done with cgal_segmentation")
    
    #----------------------now return a dictionary of the sdf values like in the older function get_sdf_dictionary
    #get the sdf values and store in sdf_labels
    sdf_labels = component_data["sdf"].tolist()
        
    sdf_temp_dict = {}
    labels_seen = []
    #iterate through the labels_list
    for i,label in enumerate(labels_list):
        if label not in labels_seen:
            labels_seen.append(label)
            sdf_temp_dict[label] = []
        
        sdf_temp_dict[label].append(sdf_labels[i])
    #print(sdf_temp_dict)
    
    #now calculate the stats on the sdf values for each label
    sdf_final_dict = {}
    for dict_key,value in sdf_temp_dict.items():
        """
        #calculate the average
        mean = np.mean(value)
        #calculate the median
        median = np.median(value)
        #calculate the max
        max = np.amax(value)
        #calculate minimum
        min = np.amin(value)
        
        temp_dict = {"mean":mean,"median":median,"max":max,"min":min}
        
        #assign them 
        sdf_final_dict[key] = temp_dict.copy()
        """
        
        #just want to store the median
        sdf_final_dict[dict_key] = np.median(value)

    return sdf_final_dict, labels_list
    
import sys
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import time


def find_neighbors(labels_list,current_label,verts_to_Face,faces_raw,verts_raw):
    """will return the number of neighbors that border the segment"""

    #iterate over each face with that label
    #   get the vertices of that face
    #   get all the faces that have that vertice associated with that
    #   get the labels of all of the neighbor faces, for each of these labels, add it to the neighbors 
    #list if it is not already there and doesn't match the label you are currently checking
    #   return the list 
    
    #get the indexes of all of the faces with that label that you want to find the neighbors for
    index_list = []
    for i,x in enumerate(labels_list):
        if x == current_label:
            index_list.append(i)
    
    verts_checked = []
    faces_checked = []
    neighbors_list = []
    neighbors_shared_vert = {}
    for index in index_list:
        current_face = faces_raw[index]
        
        #get the vertices associates with face
        vertices = current_face.vertices
        
        #get the faces associated with the vertices of that specific face
        for vert in vertices:
            #will only check each vertex once
            if vert not in verts_checked:
                verts_checked.append(vert)
                faces_associated_vert = verts_to_Face[vert]
                for fac in faces_associated_vert:
                    #make sure it is not a fellow face with the label who we are looking for the neighbors of
                    if (fac not in index_list):
                        #check to see if checked the the face already
                        if (fac not in faces_checked):
                            if(labels_list[fac] not in neighbors_list):
                                #add the vertex to the count of shared vertices
                                neighbors_shared_vert[labels_list[fac]] = 0 
                                #only store the faces that are different
                                neighbors_list.append(labels_list[fac])
                                #faces_to_check.append(fac)
                                #faces_to_check.insert(0, fac)
                            #increment the number of times we have seen that label face
                            neighbors_shared_vert[labels_list[fac]] = neighbors_shared_vert[labels_list[fac]] + 1
                            #now add the face to the checked list
                            faces_checked.append(fac)
    
    #have all of the faces to check
    
    """for facey in faces_to_check:
        if labels_list[facey] != current_label and labels_list[facey]  not in neighbors_list:
            neighbors_list.append(labels_list[facey] )"""
    
    number_of_faces = len(index_list)
    
    #can filter out the neighbors that do not have 3 or more vertices
    #print("neighbors_list = " + str(neighbors_list))
    #print("neighbors_shared_vert = " + str(neighbors_shared_vert))
    
    """final_neighbors_shared_vert = {}
    for key,value in neighbors_shared_vert.items():
        if value >= neighbors_min or key == "backbone":
            #add them to the final list if more than 3 neighbors:
            final_neighbors_shared_vert[key]= value
    
        
    final_neighbors_list = final_neighbors_shared_vert.keys()
    
    if final_neighbors_list:
        complete_Flag = True"""
        
    return neighbors_list,neighbors_shared_vert,number_of_faces


##Functins from the auto_spine_labeler
def smooth_backbone_vp3(labels_list,sdf_final_dict,backbone_width_threshold = 0.35,max_backbone_threshold = 400,backbone_threshold=300,secondary_threshold=100,shared_vert_threshold=25,number_Flag = False, seg_numbers=1,smooth_Flag=True):
    print("at beginning of smooth backbone vp3")
    #things that could hint to backbone
    #1) larger size
    #2) touching 2 or more larger size
    #have to go into object mode to do some editing
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    #print("about to get faces_verts raw")
    faces_raw = me.polygons
    verts_raw = me.vertices
    #print("DONE about to get faces_verts raw")
    
    #print("don't need to generate labels_list anymore")
    #print("about to generate labels_list")   ####!!!! This takes a good bit of time#####
    #labels_list = generate_labels_list(faces_raw)
    #print("DONE about to generate labels_list")
        
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    
    
    
    #print("about to generate verts_to_Face")
    verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
    #print("DONE about to generate verts_to_Face")
    #add new color and reassign all of the labels with those colors as the backbone label
    
    #create a list of all the labels and which ones are the biggest ones
    from collections import Counter
    
    
    myCounter = Counter(labels_list)

    spine_labels = []
    backbone_labels = []
    
    #print(" about to get counter list")
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    for label,times in myCounter.items():
        if(times >= max_backbone_threshold):
            #print(str(label) + ":" + str(times))
            backbone_labels.append(label)   
    
    for label in myCounter.keys():
        if( sdf_final_dict[label] >= backbone_width_threshold):
            #print(str(label) + ":" + str(times))
            if(myCounter[label] > backbone_threshold) and (label not in backbone_labels):
                backbone_labels.append(label)   
    #print(" DONE about to get counter list")
    
    """for lb in sdf_final_dict:
        if( sdf_final_dict[lb] >= backbone_width_threshold):
            backbone_labels.append(lb)   """
    
    
    #print("backbone_labels = " + str(backbone_labels))
    #print("hello")
    #need ot get rid of labels that don't border other backbone_labels
    to_remove = []
    
    
    
    for i in range(0,5):
        print("smoothing round " + str(i+1))
        printout_counter = 0
        counter = 0
        for bkbone in backbone_labels:
            if bkbone not in to_remove:
                neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbone,verts_to_Face,faces_raw,verts_raw)
                 
                
                #if(bkbone == "170"):
                #    print("70 nbrs = " + str(nbrs))
                
                #counts up the number of shared vertices with backbone neighbors
                backbone_count_flag = False
                neighbor_counter = 0
                total_backbone_shared_verts = 0
                for n in neighbors_list:         
                    if (n in backbone_labels) and (n not in to_remove):
                        neighbor_counter += 1
                        total_backbone_shared_verts = total_backbone_shared_verts + neighbors_shared_vert[n] 
                            
                
                #if meets requirement of shared verts then activates flag
                if (total_backbone_shared_verts > shared_vert_threshold):
                    backbone_count_flag = True
                
                
                
                '''#prevent against the split heads with 2 or 3 
                backbone_neighbor_list = neighbors_list.copy()
                backbone_neighbor_list.append(bkbone)
                other_backbone_flag = 0
                appendFlag = False
                if(backbone_count_flag == True and neighbor_counter < 4):
                    #check the other neighbor and see if the only other backbone is the current label, if so then just a split head
                    other_backbone_flag = 0
                    for n in neighbors_list:
                        if (n in backbone_labels) and (n not in to_remove):
                            neighbors_list_of_n,neighbors_shared_vert_of_n,number_of_faces_of_n = find_neighbors(labels_list,n,verts_to_Face,faces_raw,verts_raw)
                            
                            for nb in neighbors_list_of_n:
                                if (nb  in backbone_labels) and (nb not in to_remove) and (nb not in backbone_neighbor_list):
                                    backbone_neighbor_list.append(nb)
                                    other_backbone_flag += 1
                    if other_backbone_flag == 0:
                        """if printout_counter < 5:
                            #print("For backbone = " + str( bkbone))
                            #print("neighbors_list = " + str(neighbors_list))
                            #print("backbone_neighbor_list = " + str(backbone_neighbor_list))
                            
                            #print("other_backbone_flag = " + str(other_backbone_flag))
                            appendFlag = True
                            #printout_counter +=1"""
                
                if (backbone_count_flag == True and neighbor_counter < 4) and (other_backbone_flag == 0): #len(split_head_backbone_list) >= len(backbone_neighbor_list):
                    
                    
                    for bk in backbone_neighbor_list:
                        to_remove.append(bk)
                        counter += 1
                #if not backbone neighbors and/or didn't have enought shared verts then not part of the backbone

                else:
                    if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1'''
                    
                #compute the number of shared vertices and see if fits:
                if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1
            
        
        print("counter = " + str(counter))
        if counter == 0:
                print("counter caused the break")
                break
    
    
    
    
    
    #print("to remove = " + str(to_remove))
    
    print("done Analyzing big and small segments")        
    #go through and switch the label of hte 
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    print("about to rewrite the labels")
    for i in range(0,len(labels_list)):
        if labels_list[i] in backbone_labels and labels_list[i] not in to_remove:
            labels_list[i] = "backbone"
            #faces_raw[i].material_index = num_colors
            
    
    
    
    print("DONE about to rewrite the labels")
    return labels_list, verts_to_Face

#generates the stats: connections on who it is connected to), shared_verts (how many vertices it shares between it's neighbor), mesh_number (number of face for that label)
def export_connection(labels_list,label_name, verts_to_Face,outputFlag="False",file_name="None"):
    
    #print("hello from export_connection with label_name = " + str(label_name) )
    #find all the neighbors of the label
    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #print("generating list in export connections")
    #labels_list = generate_labels_list(faces_raw)
    #print("done generating list in export connections")
    
        
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    #print("about to making verts_to_Face")
    #verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
    #print("DONE about to making verts_to_Face")
    
    total_labels_list = []
    faces_checked = []
    faces_to_check = [label_name]
    
    still_checking_faces = True
        
    connections = {}
    shared_vertices = {}
    mesh_number = {}
    
    #print("about to start checking faces")
    
    #will iterate through all of the labels with the label name until find all of the neighbors (until hitting the backbone) of the label
    while still_checking_faces:
        #will exit if no more faces to check
        if not faces_to_check:
            still_checking_faces = False
            break
        
        for facey in faces_to_check:
            if facey != "backbone":
                neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,facey,verts_to_Face,faces_raw,verts_raw)
                
                
                
                #reduce the shared vertices with a face and the backbone to 0 so doesn't mess up the shared vertices percentage
                pairs = list(neighbors_shared_vert.items())
                pre_connections = [k for k,i in pairs]
                pre_shared_vertices = [i for k,i in pairs]
                
                
                
                
                if ("backbone" in pre_connections):
                    back_index = pre_connections.index("backbone")
                    pre_shared_vertices[back_index] = 0
         
                
                connections[facey] = pre_connections
                shared_vertices[facey] = pre_shared_vertices
                mesh_number[facey] = number_of_faces

                
                for neighbors in neighbors_list:
                    if (neighbors != "backbone") and (neighbors not in faces_to_check) and (neighbors not in faces_checked):
                        faces_to_check.append(neighbors)
                
                faces_to_check.remove(facey)
                faces_checked.append(facey)
        
        #append the backbone to the graph structure
        mesh_number["backbone"] = 0
    
    #print("faces_checked = " + str(faces_checked))
    #print("DONE about to start checking faces")
    
    #save off the file to an npz file
    
    
    if(outputFlag == True):
        complete_path = str("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/Automatic_Labelers/spine_graphs/"+file_name)
        
        
        
        #package up the data that would go to the database and save it locally name of the file will look something like this "4_bcelii_2018-10-01_12-12-34"
    #    np.savez("/Users/brendancelii/Google Drive/Xaq Lab/Datajoint Project/local_neurons_saved/"+segment_ID+"_"+author+"_"+
    #        date_time[0:9]+"_"+date_time[11:].replace(":","-")+".npz",segment_ID=segment_ID,author=author,
    #					date_time=date_time,vertices=vertices,triangles=triangles,edges=edges,status=status)
        np.savez(complete_path,connections=connections,shared_vertices=shared_vertices,mesh_number=mesh_number ) 
    
    return connections,shared_vertices,mesh_number
   

def classify_spine_vp2(connections,shared_vertices,mesh_number,sdf_final_dict):
    #print("inside classify_spine")
    #head_threshold = 0.15
    
    absolute_head_threshold = 30
    stub_threshold = 40
    path_threshold = 40
    
    

    #make a new dictionary to hold the final labels of the spine
    end_labels = {k:"none" for k in mesh_number.keys()}


    #only one segment so label it as a spine
    if len(connections.keys()) <= 1:
        end_labels[list(connections.keys())[0]] = "spine_one_seg"


    total_mesh_faces_outer = sum([k for i,k in mesh_number.items()])
    #print("total_mesh_faces = " + str( total_mesh_faces_outer))

    #create the graph from these
    G=nx.Graph(connections)

    endpoint_labels,shortest_paths = find_endpoints(G,mesh_number)
    
    if endpoint_labels == []:
        for jk in end_labels.keys():
            end_labels[jk] = "backbone"
            return end_labels

    #print("endpoint_labels = "+str(endpoint_labels))
    #print("shortest_paths = "+str(shortest_paths))

    #make a new dictionary to hold the final labels of the spine
    end_labels = {k:"none" for k in mesh_number.keys()}
    end_labels["backbone"] = "backbone"

    #print("end_labels at beginning")
    #print(end_labels)



    for endpoint in endpoint_labels:
        #print("at beginning of endpoint loop with label = "+ str(endpoint))
        #get the shortest path lists
        endpoint_short_paths = shortest_paths[endpoint]
        for path in endpoint_short_paths:
            path.remove("backbone")
            path_total_mesh_faces = sum([k for i,k in mesh_number.items() if i in path])
            #print("path_total_mesh_faces = "+str(path_total_mesh_faces))
            #print("at beginning of path loop with path = "+ str(path))
            travel_index = 0
            head_found = False
            label_everything_above_as_head = False
            while (head_found == False ) and travel_index < len(path):
                current_face = path[travel_index]
                sdf_guess = sdf_likely_category(current_face,travel_index,path,False,sdf_final_dict,connections,mesh_number,absolute_head_threshold)
                if  sdf_guess != "head" or mesh_number[current_face] < absolute_head_threshold:
                    #then not of any significance BUT ONLY REASSIGN IF NOT HAVE ASSIGNMENT***
                    if end_labels[current_face] == "none":
                        end_labels[current_face] = "no_significance"
                    travel_index = travel_index + 1
                else:
                    #end_labels[current_face] = "head_reg" WAIT TO ASSIGN TILL LATER
                    if "neck" != end_labels[current_face][0:4] and "spine" !=  end_labels[current_face][0:5] :   #if not already labeled as neck or spine
                        head_found = True
                        label_everything_above_as_head = True
                    else:
                        travel_index = travel_index + 1


            #print("end of first while loop, travel_index = "+ str(travel_index) + " head_found = "+ str(head_found))
            ############Added new threshold that makes it so path length can't be really small
            if travel_index < len(path):
                travel_face = path[travel_index]
            else:
                travel_face = path[travel_index-1]
                travel_index = travel_index-1
            
            if (path[travel_index] == "backbone") or ("backbone" in connections[path[travel_index]]):
                head_found = False
                label_everything_above_as_head = True
            
            if path_total_mesh_faces<path_threshold:
                head_found = False
                label_everything_above_as_head = True
            
            
            ####do the head splitting####
            #see if there are any labels that border it that also share a high percentage of faces
            if head_found == True:
                ##will return the names of the faces that have unusually high verts sharing
                split_head_labels = get_split_heads_vp2(path[travel_index],travel_index,path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold)
                #print("split_head_labels = " + str(split_head_labels))


                if len(split_head_labels) >= 2:
                    #print("adding the split head labels")
                    for split_label in split_head_labels:
                        #######may need to add in CHECK FOR ALREADY LABELED
                        if ("head" == end_labels[split_label][0:4] or end_labels[split_label] == "none"):
                            end_labels[split_label] = "head_split"
                        #else:      THINK LABELING IT AS SPINE IS NOT WHAT WE WANT
                        #    end_labels[split_label] = "spine_head_disagree_split_head"

                    label_everything_above_as_head = True
            
            
            ###if no head was found
            if head_found == False:
                #print("no head found so labeling as neck")
                #######WILL NOT OVERWRITE UNLESS LABELED AS NO SIGNIFICANCE
                for i in path: 

                    if end_labels[i] == "no_significance" or end_labels[i] == "none" or end_labels[i][0:4] == "head":
                        end_labels[i] = "neck_no_head_on_path_head_false"

                label_everything_above_as_head = False



            #print("label_everything_above_as_head = " + str(label_everything_above_as_head))
            #need to label any of those above it in the chain labeled as insignificant to heads
            if label_everything_above_as_head == True and head_found == True:
                if end_labels[travel_face] == "none":
                    #print("labeled as head reg")
                    end_labels[travel_face] = "head_reg"
                #else:               ########don't need this because don't want to overwrite already written spine neck
                    #if "head" not in end_labels[travel_index]:
                        #end_labels[travel_index] = "spine_head_disagree"


                #will label everything above it as a head and then everything below it as neck
                #####need to account for special case where not overwrite the head_split####
                if "head" == end_labels[travel_face][0:4]:
                    #print('labeling all no_significance above as head hats')
                    for i in range(0,travel_index):
                        current_label = path[i]
                        if end_labels[current_label] == "no_significance":
                            end_labels[current_label] = "head_hat"
                        else:
                            if "head" != end_labels[current_label][0:4]:
                                end_labels[current_label] = "spine_head_disagree_above_head"
                    #print('labeling all below head as necks')
                    for i in range(travel_index+1,len(path)):
                        current_label = path[i]
                        if current_label not in split_head_labels and end_labels[current_label] != "head_split":
                            end_labels[current_label] = "neck_under_head"
                else: ###not sure when this will be activated but maybe?
                    #print("head not present so labeling everything above as neck_hat")
                    for i in range(0,travel_index):
                        current_label = path[i]
                        #####need to account for special case where not overwrite the head_split####
                        if end_labels[current_label] == "no_significance":
                            end_labels[current_label] == "neck_hats_no_head"

            #print("at end of one cycle of big loop")
            #print("end_labels = " + str(end_labels))

            #what about a head being accidentally written under another head? 
            #####you should not write a head to a spine that has already been labeled as under a head
            #####you should overwrite all labels under a head as spine_under_head

    #print("outside of big loop")
    #print("end_labels = " + str(end_labels))

    #if no heads present at all label as spines
    spine_flag_no_head = False

    for face,label in end_labels.items():
        if "head" == label[0:4]:
            spine_flag_no_head = True

    if spine_flag_no_head == False:
        #print("no face detected in all of spine")
        for label_name in end_labels.keys():
            end_labels[label_name] = "spine_no_head_at_all"
            
    
    ###### TO DO: can put in a piece of logic that seekss and labels the ones we know are necks for sure based on width


    #once done all of the paths go through and label things as stubs
    if total_mesh_faces_outer < stub_threshold:
        #print("stub threshold triggered")
        for label_name in end_labels.keys():
            if "head" == end_labels[label_name][0:4]:
                end_labels[label_name] = "stub_head"

            elif "neck" == end_labels[label_name][0:4]:
                end_labels[label_name] = "stub_neck"
            else:
                end_labels[label_name] = "stub_spine"
            
    
    
    end_labels["backbone"] = "backbone"

    ###To Do: replace where look only in 1st four indexes
    return end_labels



def relabel_segments(labels_list,current_label,new_label):
    for i,x in enumerate(labels_list):
        if x == current_label:
            labels_list[i] = new_label
            
    return labels_list

def generate_verts_to_face_dictionary(faces_raw,verts_raw):
    verts_to_Face = {}
    
    #initialize the lookup dictionary as empty lists
    for pre_vertex in verts_raw:
        verts_to_Face[pre_vertex.index] = []
        
    #print(len(verts_raw))
    #print(len(verts_to_Face))
    #print(verts_to_Face[1])
    
    for face in faces_raw:
        #get the vertices
        verts = face.vertices
        #add the index to the list for each of the vertices
        for vertex in verts:
            verts_to_Face[vertex].append(face.index)
            
    return verts_to_Face

def automatic_spine_classification_vp3(labels_list,verts_to_Face,sdf_final_dict):
    
    #process of labeling
    """1) Get a list of all of the labels
    2) Iterate through the labels and for each:
        a. Get the connections, verts_shared and mesh_sizes for all labels connected to said label 
        b. Run the automatic spine classification to get the categories for each label
        c. Create a new list that stores the categories for each label processed
        d. repeat until all labels have been processed
    3) Delete all the old colors and then setup the global colors with the regular labels
    4) Change the material index for all labels based on the categorical classification"""
    
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    faces_raw = me.polygons
    verts_raw = me.vertices
    
    #labels_list = generate_labels_list(faces_raw)
    
    final_spine_labels = labels_list.copy()
    
    processed_labels = []
    
    myCounter = Counter(labels_list)
    complete_labels =  [label for label,times in myCounter.items()]
    
    head_counter = 0
    spine_counter = 0
    neck_counter = 0
    stub_counter = 0
    for i in range(0,len(complete_labels)):
        if complete_labels[i] != "backbone" and complete_labels[i] not in processed_labels:
            #print("at beginning of spine labeling loop: about to enter export connection")
            #get the conenections, shared vertices and mesh sizes for the whole spine segment in which label is connected to
            connections,shared_vertices,mesh_number = export_connection(labels_list,complete_labels[i], verts_to_Face,outputFlag="False",file_name="None")
            #print("about to send to classify spine")
            #send that graph data to the spine classifier to get labels for that
            final_labels = classify_spine_vp2(connections,shared_vertices,mesh_number,sdf_final_dict)
            #print("done classify spines")
            head_Flag = False
            spine_Flag = False
            stub_Flag = False
            neck_Flag = False
            #relabel the list accordingly
            ############could speed this up where they return the number of types of labels instead of having to search for them############
            #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")
            for key,value in final_labels.items():
                if value[0:4] == "head":
                    head_Flag = True
                if value[0:4] == "spin":
                    spine_Flag = True
                if value[0:4] == "stub":
                    stub_Flag = True
                if value[0:4] == "neck":
                    neck_Flag = True
                
                
                
                relabel_segments(final_spine_labels,key,value)
                #add them to the list of processed labels
                processed_labels.append(key)
            #print("about to find number of heads/spines/stubs/necks PLUS RELABEL AND append them to list")
                
            if head_Flag == True:
                head_counter += 1
            if spine_Flag == True:
                spine_counter += 1
            if stub_Flag == True:
                stub_counter += 1
            if neck_Flag == True:
                neck_counter += 1
            
            
   
    
    #get the indexes for the labeling from the datajoint table
    label_data = ta3.LabelKey().fetch("numeric","description")
    #print(label_data)

    label_names = label_data[1].tolist()
    label_indexes = label_data[0].tolist()
    #print(label_names)

    spine_head_index = label_indexes[label_names.index("Spine Head")]
    spine_neck_index = label_indexes[label_names.index("Spine Neck")]
    spine_reg_index = label_indexes[label_names.index("Spine")]

    
    final_faces_labels_list = np.zeros(len(faces_raw))
    final_verts_labels_list = np.zeros(len(verts_raw))

    
    #assign the colors to the faces:
    for i,fi in enumerate(final_spine_labels):
        if fi[0:4] == "head":
            #fac.material_index = 2
            final_faces_labels_list[i] = spine_head_index
        elif fi[0:4] == "neck":
            #fac.material_index = 3
            final_faces_labels_list[i] = spine_neck_index
        elif fi[0:4] == "spin":
            #fac.material_index = 4
            final_faces_labels_list[i] = spine_reg_index
        else:
            #fac.material_index = 0
            final_faces_labels_list[i] = 0
            
        #assign the vertices an index
        for vert in faces_raw[i].vertices:
            if final_verts_labels_list[vert] == 0:
                final_verts_labels_list[vert] = final_faces_labels_list[i]
        
                
        
    
    
    #create the list of labels for the vertices
    
    
    #print("DONE about to color heads")
    
    return head_counter,neck_counter, spine_counter, stub_counter, final_verts_labels_list, final_faces_labels_list


####For automatic spine labeling
def find_endpoints(G,mesh_number):
    #will first calculate all the shortest paths for each of the nodes
    
    node_list = list(G.nodes)
    if("backbone" in node_list):
        node_list.remove("backbone")
    else:
        return [],[] 
    
    shortest_paths = {}
    for node in node_list:
        shortest_paths[node] = [k for k in nx.all_shortest_paths(G,node,"backbone")]
    
    endpoints = []
    #identify the nodes that are not a subset of other nodes
    for node in node_list:
        other_nodes = [k for k in node_list if k != node ]
        not_unique = 0
        for path in shortest_paths[node]:
            not_unique_Flag = False
            for o_node in other_nodes:
                for o_shortest_path in shortest_paths[o_node]:
                    if set(path) <= set(o_shortest_path):
                        not_unique_Flag = True
                        
            if not_unique_Flag == True:
                not_unique = not_unique + 1
                
        #decide if unique endpoint
        if not_unique < len(shortest_paths[node]):   # this means there is a unique path
            
            #if not_unique != 0:
                #print(node + "-some unique and some non-unique paths for endpoint")
            endpoints.append(node)
        
    ##print(endpoints)  
    longest_paths_list = []
    for end_node in endpoints:
        longest_path = 0
        for path in shortest_paths[end_node]:
            path_length = 0
            for point in path:
                path_length = path_length + mesh_number[point]
            if path_length > longest_path:
                longest_path = path_length
        
        longest_paths_list.append((end_node,longest_path))
        
    #print(longest_paths_list)
    longest_paths_list.sort(key=lambda pair: pair[1], reverse=True)
    #print(longest_paths_list)
    ranked_endpoints = [x for x,i in longest_paths_list]
    endpoint_paths_lengths = [i for x,i in longest_paths_list]
    
    enpoint_path_list = {}
    for endpt in ranked_endpoints:
        enpoint_path_list[endpt] = shortest_paths[endpt]
        
    
    #ranked_endpoints, longest_paths_list = (list(t) for t in zip(*sorted(zip(endpoints, longest_paths_list))))
    
    
    return ranked_endpoints, enpoint_path_list 
            

def sdf_likely_category(current_label,current_index,path,head_flag,sdf_final_dict,connections,mesh_number,absolute_head_threshold):
    #width thresholding constants
    width_thresholds = {"base":0.04, "item_top_threshold":1.5} 
    #if size is smaller than the max threshold for a head then return neck
    if mesh_number[current_label] < absolute_head_threshold:
        return "neck"
    
    #get the mean, max, and median
    median_width = sdf_final_dict[current_label]
    
    #if the median is above a certain size and the total number of traingles is above a threshold then return as head
    """sdf_head_threshold = 50
    over_median_threshold  = 0.12
    if label_mesh_number > sdf_head_threshold and median > over_median_threshold:
        return "head"
    """
    
    
    neck_near_base_threshold = 0.16
    close_neck_call_threshold = 0.09
    
    #common characteristics of neck:
    #1) median width Less than neck_cuttoff_threshold
    #2) if larger item on top and that item is not a head
    #3) if larger item on top with more then 50% heads but less width
    #4) connected to backbone
    
    
    
    #1) median width Less than neck_cuttoff_threshold, return as neck
    if median_width < width_thresholds["base"]:
        return "neck"

    #2) if larger item on top and that item is not a head or #3) if larger item on top with more then 50% heads but less width
    #width_on_top = []
    #face_number_on_top = []
    
    for i in range(0,current_index):
        face_number_on_top = mesh_number[path[i]]
        width_on_top = sdf_final_dict[path[i]]
        
        if face_number_on_top > mesh_number[current_label]:
            if head_flag == False:
                return "neck"
            
            if median_width > width_thresholds["item_top_threshold"]*width_on_top:
                return "neck"
            
    #4) connected to backbone
    if "backbone" in connections[current_label]:
        return "neck"
    

    ######check for head based on if there is significantly smaller neck underneath it (because can be very close to 0.04 cuttoff sometimes
    
    #get the mean, median and max
    
    #will return head or neck
    return "head"       


def get_split_heads_vp2(current_label,current_index, path,connections,shared_vertices,mesh_number,sdf_final_dict,absolute_head_threshold):
    final_split_heads = [current_label]
    
    split_head_threshold = 0.35
    #underneath_threshold = 0.20
    
    #the only solid number threshold
    split_head_absolute_threshold = 8
    
    heads_to_check = True
    while heads_to_check:
        #1) go to the next label below it
        if(current_index < (len(path)-1)):
            next_index = current_index + 1
            next_label = path[next_index]
        
        if(next_label == "backbone"):
            #no_more_split_head_Flag = True
            break
        
        #ask if this next satisfies  1) enough shared verts?  2) SDF head possible?
        verts_sharing_index = connections[current_label].index(next_label)
        verts_sharing = shared_vertices[current_label][verts_sharing_index]
        
        #print("split share for faces " + str(current_label) + " " +str(next_label) + "="+str(verts_sharing/mesh_number[current_label]))
        sdf_guess = sdf_likely_category(next_label,next_index,path,True,sdf_final_dict,connections,mesh_number,absolute_head_threshold)
        if verts_sharing/mesh_number[current_label] > split_head_threshold and  sdf_guess == "head" and mesh_number[next_label] > split_head_absolute_threshold:
            #add next label to the list
            final_split_heads.append(next_label)
            current_index = next_index
            current_label = next_label
            
        else:
            heads_to_check = False
                     
    return final_split_heads      



##Functins from the auto_spine_labeler
def smooth_backbone_vp5(labels_list,sdf_final_dict,backbone_width_threshold = 0.35,max_backbone_threshold = 400,backbone_threshold=300,secondary_threshold=100,shared_vert_threshold=25,backbone_neighbor_min=20,number_Flag = False, seg_numbers=1,smooth_Flag=True):
    print("at beginning of smooth backbone vp4")
    #things that could hint to backbone
    #1) larger size
    #2) touching 2 or more larger size
    #have to go into object mode to do some editing
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    #print("about to get faces_verts raw")
    faces_raw = me.polygons
    verts_raw = me.vertices
    #print("DONE about to get faces_verts raw")
    
    #print("don't need to generate labels_list anymore")
    #print("about to generate labels_list")   ####!!!! This takes a good bit of time#####
    #labels_list = generate_labels_list(faces_raw)
    #print("DONE about to generate labels_list")
        
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    
    
    
    #print("about to generate verts_to_Face")
    verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
    #print("DONE about to generate verts_to_Face")
    #add new color and reassign all of the labels with those colors as the backbone label
    
    #create a list of all the labels and which ones are the biggest ones
    from collections import Counter
    
    
    myCounter = Counter(labels_list)

    spine_labels = []
    backbone_labels = []
    
    #print(" about to get counter list")
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    for label,times in myCounter.items():
        if(times >= max_backbone_threshold):
            #print(str(label) + ":" + str(times))
            backbone_labels.append(label)   
    
    for label in myCounter.keys():
        if( sdf_final_dict[label] >= backbone_width_threshold):
            #print(str(label) + ":" + str(times))
            if(myCounter[label] > backbone_threshold) and (label not in backbone_labels):
                backbone_labels.append(label)   
    #print(" DONE about to get counter list")
    
    """for lb in sdf_final_dict:
        if( sdf_final_dict[lb] >= backbone_width_threshold):
            backbone_labels.append(lb)   """
    
    
    #print("backbone_labels = " + str(backbone_labels))
    #print("hello")
    #need ot get rid of labels that don't border other backbone_labels
    to_remove = []
    
    backbone_neighbors_dict = {}
    
    for i in range(0,5):
        print("smoothing round " + str(i+1))
        printout_counter = 0
        counter = 0
        for bkbone in backbone_labels:
            if bkbone not in to_remove:
                
                #neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbone,verts_to_Face,faces_raw,verts_raw)
                
                if bkbone not in backbone_neighbors_dict.keys():
                    neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbone,verts_to_Face,faces_raw,verts_raw)
                    backbone_neighbors_dict[bkbone] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                        number_of_faces=number_of_faces)
                else:
                    neighbors_list = backbone_neighbors_dict[bkbone]["neighbors_list"]
                    neighbors_shared_vert = backbone_neighbors_dict[bkbone]["neighbors_shared_vert"]
                    number_of_faces = backbone_neighbors_dict[bkbone]["number_of_faces"]
                
                #if(bkbone == "170"):
                #    print("70 nbrs = " + str(nbrs))
                
                #counts up the number of shared vertices with backbone neighbors
                backbone_count_flag = False
                neighbor_counter = 0
                total_backbone_shared_verts = 0
                for n in neighbors_list:         
                    if (n in backbone_labels) and (n not in to_remove):
                        neighbor_counter += 1
                        total_backbone_shared_verts = total_backbone_shared_verts + neighbors_shared_vert[n] 
                            
                
                #if meets requirement of shared verts then activates flag     #not doing shared verts as a criteria
                if (total_backbone_shared_verts > shared_vert_threshold):
                    backbone_count_flag = True
                
                
                
                '''#prevent against the split heads with 2 or 3 
                backbone_neighbor_list = neighbors_list.copy()
                backbone_neighbor_list.append(bkbone)
                other_backbone_flag = 0
                appendFlag = False
                if(backbone_count_flag == True and neighbor_counter < 4):
                    #check the other neighbor and see if the only other backbone is the current label, if so then just a split head
                    other_backbone_flag = 0
                    for n in neighbors_list:
                        if (n in backbone_labels) and (n not in to_remove):
                            neighbors_list_of_n,neighbors_shared_vert_of_n,number_of_faces_of_n = find_neighbors(labels_list,n,verts_to_Face,faces_raw,verts_raw)
                            
                            for nb in neighbors_list_of_n:
                                if (nb  in backbone_labels) and (nb not in to_remove) and (nb not in backbone_neighbor_list):
                                    backbone_neighbor_list.append(nb)
                                    other_backbone_flag += 1
                    if other_backbone_flag == 0:
                        """if printout_counter < 5:
                            #print("For backbone = " + str( bkbone))
                            #print("neighbors_list = " + str(neighbors_list))
                            #print("backbone_neighbor_list = " + str(backbone_neighbor_list))
                            
                            #print("other_backbone_flag = " + str(other_backbone_flag))
                            appendFlag = True
                            #printout_counter +=1"""
                
                if (backbone_count_flag == True and neighbor_counter < 4) and (other_backbone_flag == 0): #len(split_head_backbone_list) >= len(backbone_neighbor_list):
                    
                    
                    for bk in backbone_neighbor_list:
                        to_remove.append(bk)
                        counter += 1
                #if not backbone neighbors and/or didn't have enought shared verts then not part of the backbone

                else:
                    if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1'''
                    
                #compute the number of shared vertices and see if fits:
                if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1
            
        
        print("counter = " + str(counter))
        if counter <= 3:
                print("counter caused the break")
                break
    
    
    
    #now go through and make sure no unconnected backbone segments
    
    """Pseudo-code for filtering algorithm
    1) iterate through all of the backbone labels
    2) Go get the neighbors of the backbone
    3) Add all of the neighbors who are too part of the backbone to the backbones to check list
    4) While backbone neighbor counter is less than the threshold or until list to check is empty
    5) Pop the next neighbor off the list and add it to the neighbors check list
    6) Get the neighbors of this guy
    7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
    8) continue at beginning of loop
    -- once loop breaks
    9) if the counter is below the threshold:
        Add all of values in the neighbros already checked list to the new_to_remove
    10) Use the new_backbone_labels and new_to_remove to rewrite the labels_list
    
    """
    
    new_backbone_labels = [bkbone for bkbone in backbone_labels if bkbone not in to_remove]
    new_to_remove = []
   
    for bkbonz in new_backbone_labels:
        checked_backbone_neighbors = []
        backbone_neighbors_to_check = []
        new_backbone_neighbor_counter = 0
        shared_vert_threshold = 5
        
        if bkbonz not in backbone_neighbors_dict.keys():
            neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbonz,verts_to_Face,faces_raw,verts_raw)
            backbone_neighbors_dict[bkbonz] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                number_of_faces=number_of_faces)
        else:
            neighbors_list = backbone_neighbors_dict[bkbonz]["neighbors_list"]
            neighbors_shared_vert = backbone_neighbors_dict[bkbonz]["neighbors_shared_vert"]
            number_of_faces = backbone_neighbors_dict[bkbonz]["number_of_faces"]
                    
        
        
        for bb in neighbors_list:
            if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold:
                backbone_neighbors_to_check.append(bb)
                new_backbone_neighbor_counter += 1
        
        
        #backbone_neighbors_to_check = [nb for nb in neighbors_list if nb in new_backbone_labels]
        checked_backbone_neighbors = [nb for nb in backbone_neighbors_to_check]
        #new_backbone_neighbor_counter = len(backbone_neighbors_to_check)
        #checked_backbone_neighbors = []
        
        #4) While backbone neighbor counter is less than the threshold or until list to check is empty
        while new_backbone_neighbor_counter < backbone_neighbor_min and backbone_neighbors_to_check != []:
            #5) Pop the next neighbor off the list and add it to the neighbors check list
            current_backbone = backbone_neighbors_to_check.pop(0)
            if current_backbone not in checked_backbone_neighbors:
                checked_backbone_neighbors.append(current_backbone)
            #6) Get the neighbors of this guy
            if current_backbone not in backbone_neighbors_dict.keys():
                neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,current_backbone,verts_to_Face,faces_raw,verts_raw)
                backbone_neighbors_dict[current_backbone] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                    number_of_faces=number_of_faces)
            else:
                neighbors_list = backbone_neighbors_dict[current_backbone]["neighbors_list"]
                neighbors_shared_vert = backbone_neighbors_dict[current_backbone]["neighbors_shared_vert"]
                number_of_faces = backbone_neighbors_dict[current_backbone]["number_of_faces"]
                  
            neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,current_backbone,verts_to_Face,faces_raw,verts_raw)
            #7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
            for bb in neighbors_list:
                if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold:
                    backbone_neighbors_to_check.append(bb)
                    new_backbone_neighbor_counter += 1
            
        #9) if the counter is below the threshold --> Add all of values in the neighbros already checked list to the new_to_remove
        if new_backbone_neighbor_counter < backbone_neighbor_min:
            for bz in checked_backbone_neighbors:
                if bz not in new_to_remove:
                    new_to_remove.append(bz)
    
    #print("to remove = " + str(to_remove))
    
    print("done Analyzing big and small segments")        
    #go through and switch the label of hte 
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    print("about to rewrite the labels")
    for i in range(0,len(labels_list)):
        if labels_list[i] in new_backbone_labels and labels_list[i] not in new_to_remove:
            labels_list[i] = "backbone"
            #faces_raw[i].material_index = num_colors
            
    
    
    
    print("DONE about to rewrite the labels")
    return labels_list, verts_to_Face
    



##Functins from the auto_spine_labeler
def smooth_backbone_vp4(labels_list,sdf_final_dict,backbone_width_threshold = 0.35,max_backbone_threshold = 400,backbone_threshold=300,secondary_threshold=100,shared_vert_threshold=25,backbone_neighbor_min=10,number_Flag = False, seg_numbers=1,smooth_Flag=True):
    print("at beginning of smooth backbone vp4")
    #things that could hint to backbone
    #1) larger size
    #2) touching 2 or more larger size
    #have to go into object mode to do some editing
    currentMode = bpy.context.object.mode

    bpy.ops.object.mode_set(mode='OBJECT')
    ob = bpy.context.object
    ob.update_from_editmode()
    
    #print("object_name = " + bpy.context.object.name)
    me = ob.data
    
    #print("about to get faces_verts raw")
    faces_raw = me.polygons
    verts_raw = me.vertices
    #print("DONE about to get faces_verts raw")
    
    #print("don't need to generate labels_list anymore")
    #print("about to generate labels_list")   ####!!!! This takes a good bit of time#####
    #labels_list = generate_labels_list(faces_raw)
    #print("DONE about to generate labels_list")
        
    #need to assemble a dictionary that relates vertices to faces
    #*****making into a list if the speed is too slow*******#
    
    
    
    #print("about to generate verts_to_Face")
    verts_to_Face = generate_verts_to_face_dictionary(faces_raw,verts_raw)
    #print("DONE about to generate verts_to_Face")
    #add new color and reassign all of the labels with those colors as the backbone label
    
    #create a list of all the labels and which ones are the biggest ones
    from collections import Counter
    
    
    myCounter = Counter(labels_list)

    spine_labels = []
    backbone_labels = []
    
    #print(" about to get counter list")
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    for label,times in myCounter.items():
        if(times >= max_backbone_threshold):
            #print(str(label) + ":" + str(times))
            backbone_labels.append(label)   
    
    for label in myCounter.keys():
        if( sdf_final_dict[label] >= backbone_width_threshold):
            #print(str(label) + ":" + str(times))
            if(myCounter[label] > backbone_threshold) and (label not in backbone_labels):
                backbone_labels.append(label)   
    #print(" DONE about to get counter list")
    
    """for lb in sdf_final_dict:
        if( sdf_final_dict[lb] >= backbone_width_threshold):
            backbone_labels.append(lb)   """
    
    
    #print("backbone_labels = " + str(backbone_labels))
    #print("hello")
    #need ot get rid of labels that don't border other backbone_labels
    to_remove = []
    
    backbone_neighbors_dict = {}
    
    for i in range(0,5):
        print("smoothing round " + str(i+1))
        printout_counter = 0
        counter = 0
        for bkbone in backbone_labels:
            if bkbone not in to_remove:
                
                if bkbone not in backbone_neighbors_dict.keys():
                    neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbone,verts_to_Face,faces_raw,verts_raw)
                    backbone_neighbors_dict[bkbone] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                        number_of_faces=number_of_faces)
                else:
                    neighbors_list = backbone_neighbors_dict[bkbone]["neighbors_list"]
                    neighbors_shared_vert = backbone_neighbors_dict[bkbone]["neighbors_shared_vert"]
                    number_of_faces = backbone_neighbors_dict[bkbone]["number_of_faces"]
                
                #if(bkbone == "170"):
                #    print("70 nbrs = " + str(nbrs))
                
                #counts up the number of shared vertices with backbone neighbors
                backbone_count_flag = False
                neighbor_counter = 0
                total_backbone_shared_verts = 0
                for n in neighbors_list:         
                    if (n in backbone_labels) and (n not in to_remove):
                        neighbor_counter += 1
                        total_backbone_shared_verts = total_backbone_shared_verts + neighbors_shared_vert[n] 
                            
                
                #if meets requirement of shared verts then activates flag     #not doing shared verts as a criteria
                if (total_backbone_shared_verts > shared_vert_threshold):
                    backbone_count_flag = True
                
                
                
                '''#prevent against the split heads with 2 or 3 
                backbone_neighbor_list = neighbors_list.copy()
                backbone_neighbor_list.append(bkbone)
                other_backbone_flag = 0
                appendFlag = False
                if(backbone_count_flag == True and neighbor_counter < 4):
                    #check the other neighbor and see if the only other backbone is the current label, if so then just a split head
                    other_backbone_flag = 0
                    for n in neighbors_list:
                        if (n in backbone_labels) and (n not in to_remove):
                            neighbors_list_of_n,neighbors_shared_vert_of_n,number_of_faces_of_n = find_neighbors(labels_list,n,verts_to_Face,faces_raw,verts_raw)
                            
                            for nb in neighbors_list_of_n:
                                if (nb  in backbone_labels) and (nb not in to_remove) and (nb not in backbone_neighbor_list):
                                    backbone_neighbor_list.append(nb)
                                    other_backbone_flag += 1
                    if other_backbone_flag == 0:
                        """if printout_counter < 5:
                            #print("For backbone = " + str( bkbone))
                            #print("neighbors_list = " + str(neighbors_list))
                            #print("backbone_neighbor_list = " + str(backbone_neighbor_list))
                            
                            #print("other_backbone_flag = " + str(other_backbone_flag))
                            appendFlag = True
                            #printout_counter +=1"""
                
                if (backbone_count_flag == True and neighbor_counter < 4) and (other_backbone_flag == 0): #len(split_head_backbone_list) >= len(backbone_neighbor_list):
                    
                    
                    for bk in backbone_neighbor_list:
                        to_remove.append(bk)
                        counter += 1
                #if not backbone neighbors and/or didn't have enought shared verts then not part of the backbone

                else:
                    if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1'''
                    
                #compute the number of shared vertices and see if fits:
                if neighbor_counter <= 0 or backbone_count_flag == False:
                        to_remove.append(bkbone)
                        counter += 1
            
        
        print("counter = " + str(counter))
        if counter <= 3:
            print("counter caused the break")
            break
    
    
    
    #now go through and make sure no unconnected backbone segments
    
    """Pseudo-code for filtering algorithm
    1) iterate through all of the backbone labels
    2) Go get the neighbors of the backbone
    3) Add all of the neighbors who are too part of the backbone to the backbones to check list
    4) While backbone neighbor counter is less than the threshold or until list to check is empty
    5) Pop the next neighbor off the list and add it to the neighbors check list
    6) Get the neighbors of this guy
    7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
    8) continue at beginning of loop
    -- once loop breaks
    9) if the counter is below the threshold:
        Add all of values in the neighbros already checked list to the new_to_remove
    10) Use the new_backbone_labels and new_to_remove to rewrite the labels_list
    
    """
    print("just broke out of the loop")
    
    new_backbone_labels = [bkbone for bkbone in backbone_labels if bkbone not in to_remove]
    new_to_remove = []
    skip_labels = []
    
    print("new_backbone_labels lenght = " + str(len(new_backbone_labels)))
   
    for bkbonz in new_backbone_labels:
        if bkbonz not in skip_labels:
            print("working on backbone = " + str(bkbonz))
            checked_backbone_neighbors = []
            backbone_neighbors_to_check = []
            new_backbone_neighbor_counter = 0
            shared_vert_threshold = 5
            
            if bkbonz not in backbone_neighbors_dict.keys():
                neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,bkbonz,verts_to_Face,faces_raw,verts_raw)
                backbone_neighbors_dict[bkbonz] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                    number_of_faces=number_of_faces)
            else:
                neighbors_list = backbone_neighbors_dict[bkbonz]["neighbors_list"]
                neighbors_shared_vert = backbone_neighbors_dict[bkbonz]["neighbors_shared_vert"]
                number_of_faces = backbone_neighbors_dict[bkbonz]["number_of_faces"]
            
            for bb in neighbors_list:
                if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold:
                    backbone_neighbors_to_check.append(bb)
                    new_backbone_neighbor_counter += 1
            
            
            #backbone_neighbors_to_check = [nb for nb in neighbors_list if nb in new_backbone_labels]
            checked_backbone_neighbors = [nb for nb in backbone_neighbors_to_check]
            #new_backbone_neighbor_counter = len(backbone_neighbors_to_check)
            #checked_backbone_neighbors = []
            
            #4) While backbone neighbor counter is less than the threshold or until list to check is empty
            while new_backbone_neighbor_counter < backbone_neighbor_min and backbone_neighbors_to_check != []:
                #5) Pop the next neighbor off the list and add it to the neighbors check list
                current_backbone = backbone_neighbors_to_check.pop(0)
                if current_backbone not in checked_backbone_neighbors:
                    checked_backbone_neighbors.append(current_backbone)
                #6) Get the neighbors of this guy
                if current_backbone not in backbone_neighbors_dict.keys():
                    neighbors_list,neighbors_shared_vert,number_of_faces = find_neighbors(labels_list,current_backbone,verts_to_Face,faces_raw,verts_raw)
                    backbone_neighbors_dict[current_backbone] = dict(neighbors_list=neighbors_list,neighbors_shared_vert=neighbors_shared_vert,
                        number_of_faces=number_of_faces)
                else:
                    neighbors_list = backbone_neighbors_dict[current_backbone]["neighbors_list"]
                    neighbors_shared_vert = backbone_neighbors_dict[current_backbone]["neighbors_shared_vert"]
                    number_of_faces = backbone_neighbors_dict[current_backbone]["number_of_faces"]
                
                #7) for each of neighbors that is also on the backbone BUT HASN'T BEEN CHECKED YET append them to the list to be check and update counter
                for bb in neighbors_list:
                    if (bb in new_backbone_labels) and (bb not in checked_backbone_neighbors) and (bb not in new_to_remove) and neighbors_shared_vert[bb] > shared_vert_threshold:
                        backbone_neighbors_to_check.append(bb)
                        new_backbone_neighbor_counter += 1
                
            #9) if the counter is below the threshold --> Add all of values in the neighbros already checked list to the new_to_remove
            if new_backbone_neighbor_counter < backbone_neighbor_min:
                for bz in checked_backbone_neighbors:
                    if bz not in new_to_remove:
                        new_to_remove.append(bz)
                        print("removed " + str(checked_backbone_neighbors))
            else:
                skip_labels = skip_labels + checked_backbone_neighbors
                    
            
    
    #print("to remove = " + str(to_remove))
    
    print("done Analyzing big and small segments")        
    #go through and switch the label of hte 
    #may not want to relabel until the end in order to preserve the labels in case label a big one wrong
    print("about to rewrite the labels")
    for i in range(0,len(labels_list)):
        if labels_list[i] in new_backbone_labels and labels_list[i] not in new_to_remove:
            labels_list[i] = "backbone"
            #faces_raw[i].material_index = num_colors
            
    
    
    
    print("DONE about to rewrite the labels")
    return labels_list, verts_to_Face
    







import csv
from collections import Counter
import time

#Unused function that was previously used to distribute the computational work
#but now is already accounted for by the populate method in the computed datajoint
# Create a function called "chunks" with two arguments, l and n:
"""def get_neurons_assignment(parts,index):
    #get the list of neurons from datajoint
    l = list(set(ta3.Compartment.Component().fetch("segment_id")))
    print("len(l) = " + str(len(l)))
    print(l)
    # For item i in a range that is a length of l,
    n = int(len(l)/parts)
    if len(l)/parts > n:
        n = n + 1
    print("n = "+str(n))
    storage = []
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        #print("l[i:i+n] = " + str(l[i:i+n]))
        storage.append( l[i:i+n] )
    #print(storage)
    return(storage[index])"""


ta3 = dj.create_virtual_module('ta3', 'microns_ta3')
ta3p100 = dj.create_virtual_module('ta3p100', 'microns_ta3p100')
schema = dj.schema('microns_ta3p100')
@schema
class ComponentLabel(dj.Computed):
    definition = """
    # creates the labels for the mesh table
    -> ta3p100.ComponentAutoSegment
    time_updated      :timestamp    # the time at which the component labels were updated
    ---
    n_vertices        :int unsigned #number of vertices in component
    n_triangles       :int unsigned #number of faces in component
    labeled_vertices  :longblob     #indicate which vertices are spine,spine_head,spine_neck otherwise 0
    labeled_triangles :longblob     #indicate which faces are spine,spine_head,spine_neck otherwise 0
    n_heads           :int unsigned #totals the number of heads after classification, helps for optimization
    used_version      :tinyint      #whether this component is used in the final labels or not, 0 no, 1 yes
    
   """
    
    #key_source = ta3.ComponentAutoSegment #& 'n_triangle_indices>100' & [dict(compartment_type=comp) for comp in ['Basal', 'Apical', 'Oblique', 'Dendrite']]
    
    
    def make(self, key):        
        original_start_time = time.time()    
        start_time = time.time()
        
        #neuron_ID = 579228
        #compartment_type = "Basal"
        #component_index = 2
        #clusters = 12
        #smoothness = 0.04
        
        #Apical_Basal_Oblique_default = [12,16]
        #basal_big = [16,18]
        
        neuron_ID = str(key["segment_id"])
        #component = (ta3.Compartment.Component & key).fetch1()

        component_index = key["component_index"]
        compartment_type = key["compartment_type"]
        #print("component_size = " + str(component_size))
        
        """if (compartment_type == "Basal") & (component_size > 160000):
            cluster_list = basal_big
        else:
            cluster_list = Apical_Basal_Oblique_default"""
        
        
        #for clusters in cluster_list:
        
        print("starting on cluster took--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        print(str(key["segment_id"]) + " type:" + str(key["compartment_type"]) 
            + " index:" + str(key["component_index"]) + " cluster:" + str(key["clusters"]) 
        + " smoothness:" + str(key["smoothness"]))
        
        for obj in bpy.data.objects:
            if "neuron" in obj.name:
                obj.select = True

        ob_name = load_Neuron_automatic_spine(key)
        
        object_counter = 0
        for obj in bpy.data.objects:
            if "neuron" in obj.name:
                object_counter += 1
        
        if object_counter>1:
            raise ValueError("THE NUMBER OF OBJECTS ARE MORE THAN 1")
        
        print("loading object and box--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        #what I will need to get from datajoint acces 1) sdf_final_dict 2) labels_list, might need to make the object active
        sdf_final_dict, labels_list = get_cgal_data_and_label(key,ob_name)
        if(sdf_final_dict == [] and labels_list == []):
            print("NO CGAL DATA FOR " + str(neuron_ID))
            
            # deselect all
            bpy.ops.object.select_all(action='DESELECT')

            # selection
            #for ob in bpy.data.objects
            #bpy.data.objects[ob_name].select = True
            
            for obj in bpy.data.objects:
                if "neuron" in obj.name:
                    obj.select = True
                    
                    
            # remove it
            bpy.ops.object.delete() 
            ##########should this be a return??#########
            return
        
        print("getting cgal data--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        #complete_path = "/Users/brendancelii/Google Drive/Xaq Lab/Final_Blender/saved_sdf/sdf_saved_off.npz"
        #np.savez(complete_path,labels_list=labels_list,sdf_final_dict=sdf_final_dict)
        
        
        max_backbone_threshold = 200 #the absolute size if it is greater than this then labeled as a possible backbone
        backbone_threshold=40 #if the label meets the width requirements, these are the size requirements as well in order to be considered possible backbone
        secondary_threshold=20
        shared_vert_threshold=20
        backbone_width_threshold = 0.10  #the median sdf/width value the segment has to have in order to be considered a possible backbone 
        #labels_list,verts_to_Face = smooth_backbone_vp3(labels_list,sdf_final_dict,backbone_width_threshold,max_backbone_threshold = max_backbone_threshold,backbone_threshold=backbone_threshold
        #        ,secondary_threshold=secondary_threshold,shared_vert_threshold=shared_vert_threshold,number_Flag = False, seg_numbers=1,smooth_Flag=True)
        
        backbone_neighbor_min=20
        labels_list,verts_to_Face = smooth_backbone_vp4(labels_list,sdf_final_dict,backbone_width_threshold,max_backbone_threshold = max_backbone_threshold,backbone_threshold=backbone_threshold
                ,secondary_threshold=secondary_threshold,shared_vert_threshold=shared_vert_threshold,backbone_neighbor_min=backbone_neighbor_min,
                   number_Flag = False, seg_numbers=1,smooth_Flag=True)
                
        
        print("smoothing backbone--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        #save off the sdf value for testing:
        #save off the faces_raw as an npz file
        
        #complete_path = "/Users/brendancelii/Google Drive/Xaq Lab/Final_Blender/saved_sdf/sdf_saved_off.npz"
        #np.savez(complete_path,labels_list=labels_list,sdf_final_dict=sdf_final_dict)
        
        object_counter = 0
        for obj in bpy.data.objects:
            if "neuron" in obj.name:
                object_counter += 1
        
        if object_counter>1:
            raise ValueError("THE NUMBER OF OBJECTS ARE MORE THAN 1")
        
        head_counter,neck_counter, spine_counter, stub_counter,final_verts_labels_list, final_faces_labels_list = automatic_spine_classification_vp3(labels_list,verts_to_Face,sdf_final_dict)
        print("classifying spine--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        print("head_counter = " + str(head_counter))
        print("neck_counter = " + str(neck_counter))
        print("spine_counter = " + str(spine_counter))
        print("stub_counter = " + str(stub_counter))
        
        #now send out the labels to the table
        #now write them to the datajoint table  
        comp_dict = dict(key,
                            time_updated = str(datetime.datetime.now())[0:19],
                            n_vertices = len(final_verts_labels_list),
                            n_triangles = len(final_faces_labels_list),
                            labeled_vertices = final_verts_labels_list,
                            labeled_triangles = final_faces_labels_list,
                            n_heads = head_counter,
                            used_version = 1)


        self.insert1(comp_dict)
        print("writing label data to datajoint--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        #delete the object after this
        #delete the object
                    
        # deselect all
        bpy.ops.object.select_all(action='DESELECT')

        # selection
        #for ob in bpy.data.objects
        #bpy.data.objects[ob_name].select = True
        
        for obj in bpy.data.objects:
            if "neuron" in obj.name:
                obj.select = True
                
                
        # remove it
        bpy.ops.object.delete()
        
        print("deleting object--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        # deselect all
        bpy.ops.object.select_all(action='DESELECT')

        # selection
        #for ob in bpy.data.objects
        #bpy.data.objects[ob_name].select = True
            
        
        object_counter = 0
        for obj in bpy.data.objects:
            if "neuron" in obj.name:
                object_counter += 1
        
        if object_counter>1:
            raise ValueError("THE NUMBER OF OBJECTS ARE MORE THAN 1")
                
                    
                    
                
        
        print("finished")
        print("--- %s seconds ---" % (time.time() - original_start_time))

populate_start = time.time()
ComponentLabel.populate(reserve_jobs=True)
print("\npopulate:", time.time() - populate_start)
