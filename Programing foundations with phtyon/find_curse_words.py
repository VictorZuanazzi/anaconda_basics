# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 12:28:30 2018

@author: victzuan
"""
import urllib
from profanity import profanity

def check_profanity(text_to_check):
    is_it_dirty = profanity.contains_profanity(text_to_check)
    if is_it_dirty:
        suggested_text = profanity.censor(text_to_check)
        print("\nThe following text was censored:\n")
        print(suggested_text)
    
    return is_it_dirty
    
##not ready yet:
#def talk_like_pirates(text_to_check):
#    #The website is not available:
#    connection = urllib.request.urlopen("http://isithackday.com/arrpi.php?text="+ text_to_check)
#    output = connection.read()
#    print(output)
#    connection.close()

def read_text():
    
    quotes = open("C:\\Users\\victzuan\\.spyder-py3\\Programing foundations with phtyon\\no curse words.txt")
    contents_of_file = quotes.read()
    print(contents_of_file)
    quotes.close()
    if check_profanity(contents_of_file):
        print('\nConsider reviewing your text\n')
    else:
        print('\nYour text is fine! =)\n')
    
read_text()
