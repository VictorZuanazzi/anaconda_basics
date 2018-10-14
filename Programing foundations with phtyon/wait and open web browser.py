# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:23:44 2018

@author: victzuan
"""

import webbrowser
import time

total_breaks = 3

print ("This program started on " + time.citime())
for i in range(total_breaks):
    #Step1. Count down X minutes or hours.
    time.sleep(2*60*60)

    #Step2. Prompt a break message or do something cool.
    webbrowser.open("https://www.facebook.com/messages/t/tran.linh.ph")
