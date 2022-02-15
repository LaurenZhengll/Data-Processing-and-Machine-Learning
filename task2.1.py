# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:19:58 2020

@author: laure
"""
#define a function to recerive a nonnegative integar and calculate its factorial
def factorial(a):
    print("Factorial of {0}:", a)
    # 1!=1, 0!=1
    if a == 0 or a == 1:
        print("1")
    else:
        # divide factorial into two multiplied natural integar
        c = 1
        while a > 1:
            b = a * (a - 1)
            c = c * b
            a -= 2
        print(c)

while True:
    a = int(input("Please input a nonnegative integar:"))
    if a < 0:
        # input must >= 0
        print("Sorry, input must be a nonnegative integar")   
    else:
        #call the factorial function and jump out of while loop
        factorial(a)
        break
    
    

    