# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:28:02 2018

@author: victzuan
"""

class Parent():
    def __init__(self, last_name, eye_color):
        print('Parent Contructor Called')
        self.last_name = last_name
        self.eye_color = eye_color
    
    def show_info(self):
        print("Last Name: " + self.last_name +"\nEye Color: "+ self.eye_color)
        

class Child(Parent):
    def __init__(self, last_name, eye_color, number_of_toys):
        print ("Child Contructor Called")
        Parent.__init__(self, last_name, eye_color)
        self.number_of_toys = number_of_toys
            
        
billy_cyrus = Parent("Cyrus", "blue")
billy_cyrus.show_info()

miley_cyrus = Child('Cyrus', 'Blue', 5)
miley_cyrus.show_info()
#print(miley_cyrus.last_name)
print(miley_cyrus.number_of_toys)
#print(miley_cyrus.eye_color)