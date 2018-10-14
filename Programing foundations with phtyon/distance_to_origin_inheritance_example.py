# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 08:35:17 2018

@author: victzuan
"""
import math

#Example of Inheritance:

class Point:
    def __init__(self, x=0, y=0):
        #print('Point')
        self.x = x
        self.y = y
        
    def print_point(self):
        print("x: ", self.x, "y: ",self.y)
    
    def print_distance_to_origin(self):
        print("Distance to origin: ", self.distance_to_origin())
    
    def print_distance_from_point(self, point):
        print("Distance from point (",point.x, ",", point.y,"):", self.distance_from_point(point))
    
        
class Cartesian(Point):
    def __init__(self, x, y):
        #print ('Cartesian')
        Point.__init__(self, x, y)
    
    def distance_to_origin(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def distance_from_point(self, point):
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

class Manhattan(Point):
    def __init__(self, x, y):
        #print ('Manhattan')
        Point.__init__(self, x, y)
        
    def distance_to_origin(self):
        return self.x + self.y

    def distance_from_point(self, point):
        return (math.fabs(self.x - point.x) + math.fabs(self.y - point.y))

z = Cartesian(3,4)
p = Manhattan(4,3)

z.print_point()
z.print_distance_to_origin()
z.print_distance_from_point(p)

p.print_point()
p.print_distance_to_origin()
p.print_distance_from_point(z)
