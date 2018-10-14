# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 07:32:13 2018

@author: victzuan
"""
import os

def create_a_message(message = 'such a sunny day.'):
    '''
    '''
    folder_path = input('please paste the folder adress:\n')
    
    program_path = os.getcwd()
    
    order = 'abcdefghijklmnopqrstuvwxyz'
    
    #Save the new files in the same directory:
    os.chdir(folder_path)
    for letter in message:
        new_name = letter+'.png'
        os.rename(file_name,new_name)
        

create_a_message()