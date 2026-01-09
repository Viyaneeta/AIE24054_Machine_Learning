#Machine Learning - Lab Assignment 1
#Viyaneeta Ramesh - BL.SC.U4AIE24054
#Multiply two matrices

def matrix_multiplication(A, B, A_rows, A_cols, B_cols):
    result=[]  #result matrix with zeros
    for i in range(A_rows):
        row=[]
        for j in range(B_cols):
            row.append(0)
        result.append(row)

    #matrix multiplication
    for i in range(A_rows):           #loop through rows of A
        for j in range(B_cols):       #loop through columns of B 
            for k in range(A_cols):   #loop through columns of A or rows of B
                result[i][j]=result[i][j]+A[i][k]*B[k][j]
    return result


def main():
    A = []
    B = []

    #Input
    A_rows=int(input("Enter no. rows in Matrix A: "))
    A_cols=int(input("Enter no. columns in Matrix A: "))
    B_rows=int(input("Enter no. of rows in Matrix B: "))
    B_cols=int(input("Enter no. of columns in Matrix B: "))

    #multiplication condition
    if A_cols!=B_rows:
        print("Matrix Multiplication not possible")
        return

    #matrix A
    print("Enter elements of Matrix A:")
    for i in range(A_rows):
        A.append([])
        for j in range(A_cols):
            x=int(input("Enter:"))
            A[i].append(x)

    #matrix B
    print("Enter elements of Matrix B:")
    for i in range(B_rows):
        B.append([])
        for j in range(B_cols):
            x=int(input("Enter element:"))
            B[i].append(x)

    #multiply
    product=matrix_multiplication(A, B, A_rows, A_cols, B_cols)
    print("Product matrix:", product)


main()

