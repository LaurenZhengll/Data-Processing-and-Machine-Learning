# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:07:05 2020

@author: laure
"""

#define a recusive function to recerive a nonnegative integar and calculate its factorial
def factorial(a):   
    if a > 1:
        # 5! = 5 * 4!
        b = a * factorial(a - 1)      
    else:
        # 1!=1, 0!=1
        b = 1
    return b
        

while True:
    a = int(input("Please input a nonnegative integar:"))
    if a < 0:
        # input must >= 0
        print("Sorry, input must be a nonnegative integar")   
    else:
        print("Factorial of {0}:", a)
        #call the recursive factorial function and jump out of while loop
        b = factorial(a)
        print(b)
        break
