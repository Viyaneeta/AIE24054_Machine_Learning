#Machine Learning - Lab Assignment 1
#Viyaneeta Ramesh - BL.SC.U4AIE24054
#count of common elements in two lists

def common_elements_count(list1,list2):
    count=0
    
    for i in list1:
        if i in  list2:       #i in list1 and list2, therefore common
            count+=1
    return count

#intiliaze lists
list1=[]
list2=[]

#input list1
n1=int(input("Enter number of elements in first list: "))
for i in range(n1):
    x=int(input("Enter element: "))
    list1.append(x)
print(list1)
    
#input list2
n2=int(input("Enter number of elements in second list: "))
for i in range(n2):
    y=int(input("Enter element: "))
    list2.append(y)
print(list2)

res=common_elements_count(list1,list2)   #function call
print("Number of common elements:", res)


