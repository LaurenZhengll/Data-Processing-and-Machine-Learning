# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:56:39 2020

@author: laure
"""
#get width from user
width = int(input("Please input an integer as the rect width:"))
#width should be greater than 0
if width <= 0:
    print("Sorry, width should be greater than 0")
else:
    #get height from user
    height = int(input("Please input an integer as the rect height:"))
    #height should be greater than 0
    if height <= 0:
        print("Sorry, height should be greater than 0")
    else:
        #iterate to output the rect
        x = 1
        y = 1        
        while y <= height:
            while x <= width:
                #output without line break
                print('* ',end='')
                x += 1
            x = 1
            y += 1
            #output with line break
            print('\n')
         
""" output using for loop      
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                print('* ',end='')
                x += 1
            x = 1
            y += 1
            print('')
"""