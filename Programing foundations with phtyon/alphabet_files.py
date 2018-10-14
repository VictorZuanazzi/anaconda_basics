# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 07:21:50 2018

@author: victzuan
"""

import os

def alphabet_files():
    '''rename a set of files to the letters of the alphabet.
    '''
    #1. Get file names from a folder.
    folder_path = input('please paste the folder adress:\n')
    file_list = os.listdir(folder_path)
    program_path = os.getcwd()
    #print (file_list)
    
    #Save the new files in the same directory:
    os.chdir(folder_path)
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz. '
    
    #2. For each file, rename filename.
    for i in range(len(file_list)):
        file_name = file_list[i]
        letter = alphabet[i] 
        new_name = letter+'.png'
        os.rename(file_name,new_name)
    #Save the new files in the same directory:
    os.chdir(program_path)
        
    
    
alphabet_files()
