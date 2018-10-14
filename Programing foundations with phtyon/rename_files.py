# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 05:59:50 2018

@author: victzuan
"""
import os

def rename_files():
    '''take numbers from the name of files.
    '''
    #1. Get file names from a folder.
    folder_path = input('please paste the folder adress:\n')
    file_list = os.listdir(folder_path)
    program_path = os.getcwd()
    #print (file_list)
    
    #Save the new files in the same directory:
    os.chdir(folder_path)
    
    #2. For each file, rename filename.
    for file_name in file_list:
        remove_digits = str.maketrans('','', '0123456789')
        new_name = file_name.translate(remove_digits)
        print ("Converting file: " + file_name +"into "+ new_name + " ... ")
        os.rename(file_name,new_name)
    #Save the new files in the same directory:
    os.chdir(program_path)
        
    
    
rename_files()