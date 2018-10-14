# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:16:29 2018

@author: victzuan
"""

from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "AC0251a23211a813fb79ce9d8cee3c9eb7"
# Your Auth Token from twilio.com/console
auth_token  = "85e99e66469b8f8991c3e8baadcc358e"

client = Client(account_sid, auth_token)

message = client.messages.create(
    #to="+32497844660", #My phone!
    to="+32478425318", #Tess' phone
    from_="+32460204787",
    body="Did I tell you today how sexy you are?")

print(message.sid)