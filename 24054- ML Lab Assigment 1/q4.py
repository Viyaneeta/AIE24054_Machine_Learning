#Machine Learning - Lab Assignment 1
#Viyaneeta Ramesh - BL.SC.U4AIE24054
#find the tranpose of the matrix

def find_transpose(A, rows, columns):
    transpose=[]

    for i in range(columns):       #matrix column becomes tranpose row
        trow=[]
        for j in range(rows):
            trow.append(A[j][i])    #swap row and column
        transpose.append(trow)
        
    return transpose

A=[]

#input dimensions
rows=int(input("Enter number of rows: "))
columns=int(input("Enter number of columns: "))

#input matrix
for i in range(rows):
    A.append([])
    for j in range(columns):
        x=int(input("Enter element: "))
        A[i].append(x)
        
transpose=find_transpose(A,rows,columns)

for i in transpose:
    print(i)
