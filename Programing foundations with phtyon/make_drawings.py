# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:25:57 2018

@author: victzuan
"""
import turtle

def set_up_window(background = "white"):
    '''set up the back ground window.
    '''
    window = turtle.Screen()
    window.bgcolor(background)
    return window

def make_triangle_eq(artist, L = 100, fill = False):
    '''draws a triangle with all sizes L.
        
        args:
            artist: instance of class Turtle.
            L: the lenght of one side of the triangle.
    '''
    if fill:
        artist.begin_fill()
    
    for i in range(3):
        artist.forward(L)
        artist.left(180-60)
        
    if fill:
        artist.end_fill()

def draw_eq_flower():
    '''draws several squares forming a circle-like shape.
    '''
    mss_brush = turtle.Turtle()
    mss_brush.shape("turtle")
    mss_brush.color("orange")
    mss_brush.speed()

    angle = 45
    mss_brush.left(90)
    for j in range(360//angle):
        make_triangle_eq(mss_brush, L = 100)
        mss_brush.left(angle)
    
    mss_brush.color("green")
    mss_brush.right(180)
    mss_brush.forward(150)
    make_triangle_eq(mss_brush, fill = True)
    mss_brush.forward(150)
    mss_brush.right(90)
    mss_brush.forward(100)
    mss_brush.right(180)
    mss_brush.forward(200)
    
def triangle_fractal():
    mss_brush = turtle.Turtle()
    mss_brush.shape("turtle")
    mss_brush.color("orange", "green")
    mss_brush.speed()
    angle = 60
    side_size = 400
    mss_brush.left(90)
    for j in range(720//angle):
        side_size = side_size/(1+j/10)
        print(side_size)
        if j%2 ==0:
            full_fill = True
        else:
            full_fill = False
        make_triangle_eq(mss_brush, L = side_size, fill = full_fill)
        mss_brush.left(angle)
    

def draw_circle(D = 100):
    '''draws a cirlce of diameter D.
    '''
    angie = turtle.Turtle()
    angie.shape("turtle")
    angie.color("blue", "grey")
    angie.goto(0,-100)
    angie.begin_fill()
    angie.circle(D)
    angie.end_fill()

def draw_triangle():
    '''draws a triangle equilatero.
    '''
    mr_brush = turtle.Turtle()
    mr_brush.color('gray')
    mr_brush.shape('turtle')
    mr_brush.forward(100)
    
    for i in range(3):
        mr_brush.right(180-60)
        mr_brush.forward(200)

window= set_up_window()    
#draw_circle()
#draw_triangle()
#draw_squares()

draw_eq_flower()

#triangle_fractal()
window.exitonclick()