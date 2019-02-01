#!/usr/bin/env python
# coding: utf-8

# In[10]:


'''
Purpose: Will generate the extra python script for running the full neuron classification

For the range specified (ex: 40/50)
- Create a copy of the python file
- generate a new name based on the iterator
- rename the file
- collect the name in a list

Open up a new bash file
1) write #1/bin/bash
2) For each name in the list: write "python [name] &" as a line
3) make it into an executable

'''


# In[11]:


#generate the copies
from shutil import copyfile

number_files = 50
file_name = "decimation_script_orphans.py"
total_file_lists = list()
total_file_lists.append(file_name)
for i in range(0,number_files):
    #create new name for file
    new_name = file_name[:-3] +"_"+ str(i) + ".py"
    #copy the file
    copyfile(file_name, new_name)
    #add to the total list
    total_file_lists.append(new_name)
    
    
    


# In[18]:


#create theh bash file for the list
filename = "decimation_orphans.sh"
f = open(filename, "w")
f.write("#!/bin/bash\n")
for ll in total_file_lists:
    f.write("python " + ll + " &\n")
f.close()


# In[19]:


#how to compile the bash script
#import os
#os.system("chmod +x "+filename)


# In[17]:


# #how to clean up all of the files once done:
# #create theh bash file for the list
# import os
# for ll in total_file_lists[1:]:
#     os.remove(ll)


# In[ ]:




