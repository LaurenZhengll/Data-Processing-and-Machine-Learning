# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:32:46 2020

@author: laure
"""
import re

def emailValidation(email):
    list1 = re.findall('@', email)
    # one and only one @
    if len(list1) != 1:
        print("This is not a valid email!")
        return -1
    else:
        """valid format: abc@def.gh abc@def.gh.ij abc.ef@gh.ij abc.ef@gh.ij.kl
         abc.-ef@gh.ij -@gh.ij -a@ij.ij a-@ij.ij .@i.ij  a@ij..ij a@i.i. a@.i.i. a@-ij.ij a@ij-.ij
         a@i.jm-.km a@i.j.k a@0.im a.b.c.d.--@i.j.k.l
         invalid format: a@ij.-ij a@i.j a@i.j0 a@i.0j @i.jm a@.im a?@i.ij a@ij.ij-
         """
        #[abc]: existing a or b or c; \w: 0-9a-zA-Z_; *: >=0; +: >=1 
        result = re.match("[\w*.*-*]+@\w+(.[a-zA-Z][a-zA-Z]+)+",email)
        if result:
            print("valid email")
            list2 = re.split('@',email)
            # result.string: the string passed into the function
            print("email: " + result.string + ',' + "username: " + list2[0] + ',' + "host: "+ list2[1])
            return 1
        else:
            print("invalid email")
            return -1
            
while True:
    email = input("Please intput your email address:")
    y = emailValidation(email)
    if y == 1:break
       
    
    
    
    
    

       