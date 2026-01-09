#Machine Learning - Lab Assignment 1
#Viyaneeta Ramesh - BL.SC.U4AIE24054
#find mean, median and mode

import random

def generate_random():
    num=[]
    for i in range(100):
        num.append(random.randint(100,150))
    return num

def find_mean(num):
    sum=0
    for i in num:
        sum+=i
    return sum/len(num)   #total sum of values/ number of values

def find_median(num):
    num.sort()
    mid=len(num)//2
    median=(num[mid-1]+num[mid])/2  #if two middle numbers, add them and divide by 2
    return median

def find_mode(num):
    dict1={}                 #dictionary to store frequency of each element
    for i in num:
        if i in dict1:
            dict1[i]+=1      #increment if element exists
        else:
            dict1[i]=1       #first appearance 

    max1=0
    mode=num[0]

    for x in dict1:
        if dict1[x]>max1:
            max1=dict1[x]
            mode=x
    return mode

nums=generate_random()
print("Mean:",find_mean(nums))
print("Median:",find_median(nums))
print("Mode:",find_mode(nums))
