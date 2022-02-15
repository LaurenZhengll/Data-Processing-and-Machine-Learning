# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

#create a list
mylist = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

#print type
print("The type:",type(mylist))

#iterate and print all items of the list
for x in mylist:
    print(x)


#create a tuple
mytuple = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")

#print type
print("The type:",type(mytuple))

#iterate and print all items of the tuple
for x in mytuple:
    print(x)
    

#create a set
myset = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}

#print type
print("The type:",type(myset))

#iterate and print all items of the set
for x in myset:
    print(x)


#create a dictionary
mydict = {"Monday": 1,
         "Tuesday": 2,
         "Wednesday": 3,
         "Thursday": 4,
         "Friday": 5,
         "Saturday": 6,
         "Sunday":7}

#print type
print("The type:",type(mydict))

#iterate and print all items of the dictionary
for x,y in mydict.items():
    print(x,y)
