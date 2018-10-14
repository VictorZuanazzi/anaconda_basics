# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:24:56 2018

@author: victzuan
"""

import urllib

def read_website():
    connection = urllib.request.urlopen("http://www.pudim.com.br/")
    output = connection.read()
    
    print(output)
    connection.close()

read_website()