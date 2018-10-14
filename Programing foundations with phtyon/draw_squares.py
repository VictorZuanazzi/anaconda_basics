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

def draw_squares():
    '''draws several squares forming a circle-like shape.
    '''
    mss_brush = turtle.Turtle()
    mss_brush.shape("turtle")
    mss_brush.color("black")
    mss_brush.speed()

    angle = 5
    mss_brush.left(90)
    for j in range(360//angle):
        for i in range(4):
            mss_brush.forward(100)
            mss_brush.right(90)
        mss_brush.left(angle)
        if j%2 == 0:
            mss_brush.color("yellow")
        else:
            mss_brush.color("black")

def draw_circle(D = 100):
    '''draws a cirlce of diameter D.
    '''
    angie = turtle.Turtle()
    angie.shape("turtle")
    angie.color("blue")
    angie.goto(0,-100)
    angie.circle(D)

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

#set_up_window()    
#draw_circle()
#draw_triangle()
#draw_squares()
#
#window.exitonclick()