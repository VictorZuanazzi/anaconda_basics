# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:44:47 2018

@author: victzuan
"""

#The functions were defined separetly so we can just call the wanted example
#whithout worring about the inputs and outputs of other functions.
#However, it should be highligheted that the lambda is used to indeed avoid 
#defining some sort of functions.

#1st example:
def lambda_sum(x, y):
    '''(number, number) -> number
        (list, list) -> list
        (tuple, tuple) -> tuple
        (str, str) -> str
    
        args:
            x, y: numbers that to be summed.
            x, y: lists, tuples or strings to be concatenated.
        return:
            return the sum of the two numbers.
            return the concatenation of the list, tuples or strings.       
    >>> lambda_sum(3,4)
    7
    >>> lambda_sum([1,2], [3,4])
    [1,2,3,4]
    >>> lambda_sum("ceci n'est pas ", "un pipe")
    "ceci n'est pas un pipe"
    
    '''
    l_sum = lambda a,b : a + b
    return l_sum(x,y)

#2nd example, map() function:
#map() is a function which takes two arguments: 
#
#r = map(func, seq)
#
#The first argument func is the name of a function and the second a sequence
#(e.g. a list) seq. map() applies the function func to all the elements of the
#sequence seq. Before Python3, map() used to return a list, where each element 
#of the result list was the result of the function func applied on the 
#corresponding element of the list or tuple "seq". With Python 3, map() returns 
#an iterator.
    
def temperature_map(choice, temperatures):
    
    #The two functions return the same values, but one has its functions
    #defined explecitely and the other uses lambda for that.
    
    def no_lambda(temperatures):
        
        def fahrenheit(T):
            '''return the temperature in farenheit.
            
            args:
                T: number or list of numbers with temperature in celcius.
            retrun number or list of numbers temperature in farenheit.
            '''
        
            return (9*T)/5 + 32
        
        def celcius(T):
            '''return the temperature in celcius.'''
            return (5/9)*(T-32)
        
        temperatures_in_Fahrenheit = list(map(fahrenheit, temperatures))
        temperatures_in_Celcius = list(map(celcius, temperatures_in_Fahrenheit))
        
        return temperatures_in_Fahrenheit, temperatures_in_Celcius
    
    def with_lambda(temperatures):
        temperatures_in_Fahrenheit = list(map(lambda x: 9*x/5 + 32, temperatures))
        temperatures_in_Celcius = list(map(lambda x: (5/9)*(x-32), temperatures_in_Fahrenheit))
        return temperatures_in_Fahrenheit, temperatures_in_Celcius
    
    if choice == "no_lambda":
        return no_lambda(temperatures)
    elif choice == "with_lambda":
        return with_lambda(temperatures)
    else:
        return print("that is weird... " + choice + " is not a valid instruction\nonly *no_lambda(temperatures)* and *with_lambda(temperatures)* are valid")
    

#map() can also be used with to iterate a list of functions.  
from math import sin, cos, tan, pi
    
def iterate_function(choice, family_of_functions = (sin, cos, tan), value= pi):
    '''execute all functions given at family_of_functions with the input value.
    
    args:
        family_of_functions: list of functions
        value: input to the functions in family_of_functions
    
    >>> iterate_function(1)
    [1.2246467991473532e-16, -1.0, -1.2246467991473532e-16]
    >>> iterate_function(2)
    [1.2246467991473532e-16, -1.0, -1.2246467991473532e-16]
    '''
    #map_functions_1 and map_functions_2 are just different implementations,
    #both return the same valules.
    
    def map_functions_1(x, functions):
        """ map an iterable of functions on the the object x """
        
        res = []
        for func in functions:
            res.append(func(x))
        return res
    
    def map_functions_2(x, functions):
        """ map an iterable of functions on the the object x """
        return [func(x) for func in functions]
    
    if choice == 1:
        return map_functions_1(value, family_of_functions)
    elif choice == 2:
        return map_functions_2(value, family_of_functions)
    else:
        print ("you don't know what you want, do you?\n you can chose only 1 or 2. ", choice, " is not an option")
    
# 3rd example, filter():
# filter() offers an elegant way to filter out all the elements of a sequence 
#"sequence", for which the function function returns True. i.e. an item will be
#produced by the iterator result of filter(function, sequence) if item is 
#included in the sequence "sequence" and if function(item) returns True.

def create_fibonacci_list(number_of_elements):
    '''return a list of len(number_of_elements) in fibonacci sequence starting
    from 0.
    
    >>> create_fibonacci_list(2):
    [0, 1]
    '''
    if number_of_elements <= 0:
        return []
    elif number_of_elements == 1:
        return [0]
    else:
        fibo = [0,1]
        for i in range(2, number_of_elements):
            fibo.append(fibo[i-1] + fibo[i-2])
        return fibo

def fibonacci_lambda(number_of_elements):
    ''' (number) -> list of number
    
        creates a list of len(number_of_elements) with fibonnaci numbers.
        The list starts from 0
        
        args:
            number_of_elements: the length of the list.
        return:
            list with fibonacci numbers.
    ''' 
    # t is a sequential list with elements from 0 to a.
    t = (lambda a: list(range(0,a))) 
    # fib_n is the nth fibonacci number.
    fib_n = lambda n, x=0, y=1 : x if not n else fib_n(n-1, y, x+y)
    # creats a list of len(number_of_elements) of fibonacci numbers.
    fib = list(map(fib_n, t(number_of_elements)))
    return fib


def filter_odds(a_list = create_fibonacci_list(11)):
    '''return only the the odd numbers of a list.
    '''
    odd_numbers = list(filter(lambda x: x%2, a_list)) #x%2 returns 0 or 1 that is interpreted as 0: false, 1: true.
    return odd_numbers
        
def filter_evens(a_list = create_fibonacci_list(11)):
    '''return only the the even numbers of a list.
    '''
    even_numbers = list(filter(lambda x: x%2 == 0, a_list))
    return even_numbers

#4th example, reduce():
    
import functools

def sum_reduce(a_list = create_fibonacci_list(5)):
    '''returns the sum of a list of numbers or the concatenation of lists, 
    tuples and strings'''
    #reduced = functools.reduce(lambda x,y : x+y, a_list)
    reduced = functools.reduce(lambda x,y : lambda_sum(x,y), a_list)
    return reduced

def max_reduce(a_list = create_fibonacci_list(5)):
    '''return the max value of a list'''
    f = lambda a, b: a if (a>b) else b
    max_value = functools.reduce(f, a_list)
    return max_value

def factorial(number = 10):
    '''return the factorial of the given number
    '''
    if number >= 0:
        return 1
    else: 
        fac = functools.reduce(lambda x, y: x*y, range(1, number))
        return fac


def filter_elements(list_of_elements, string_of_interest, contain = True):
    '''filsters a list of strings based on a string of interest. 
    
    Returns:
        if cointain == True, returns all elements in which the string of 
            interest is present.
        if contains == False, returns all elements in which the string of 
            interest is not present.
    
    >>> l=  ['potato', 'positive', 'pasta']
    >>> filter_elements(l, 'po')
    >>> ['potato', 'positive']
    >>> filter_elements(l, 'ta', contain = False)
    >>> ['positive']
    '''
    if contain:
        p = list(filter(lambda x: string_of_interest in x, list_of_elements))
    else:
        p = list(filter(lambda x: not string_of_interest in x, list_of_elements))
    return p












