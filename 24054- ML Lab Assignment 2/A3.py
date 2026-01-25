# LAB 2 - Q3


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def mean_variance(data):
    mean1=np.mean(data)       #calculation of mean and variance using numpy import and functions
    var1=np.var(data)
    return mean1,var1


def calc_mean(data):
    mean2=sum(data)/len(data)        #mean using formula (total sum/no. of items)
    return mean2

def calc_variance(data):
    m=calc_mean(data)
    var2=sum((x-m)**2 for x in data)/len(data)     #variance using formula sigma^2/len(data)

    return var2

def running_time(func,data,runs=10):
    total=0
    for _ in range(runs):
        start=time.perf_counter()
        func(data)
        total=total+time.perf_counter()-start      #run time comparison

    return total/runs



def loss_prob(change_column):
    return (change_column.apply(lambda x: x < 0)).mean()         #probability of loss over the stock (used lambda method)


def wednesday_profit(df):
    wed_data=df[df['Day'] == 'Wed']                      #probability of profit on wednesday 
    return(wed_data['Chg%'] > 0).mean()

def main():
    df=pd.read_excel(r"C:\Users\Viyaneeta Ramesh\OneDrive\Documents\SEM 4\Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
    price=df['Price']

    x,y=mean_variance(price)
    print("Numpy mean:",x)
    print("Numpy variance:",y)    #print numpy mean and variance

    u=calc_mean(price)
    v=calc_variance(price)        # print custom mean and variance
    print("Mean:",u)
    print("Variance:",v)

    print("Average time for Numpy mean:",running_time(np.mean,price))
    print("Average time for Numpy variance:",running_time(np.var,price))              #compare running time 
    print("Average time for Normal mean:",running_time(calc_mean,price))
    print("Average time for Normal variance:",running_time(calc_variance,price))


    wed_price=df[df['Day']=='Wed']['Price']
    wed_mean=np.mean(wed_price)                 #sample mean for all wednesdays
    print("Wednesday mean:",wed_mean)

    apr_price=df[df['Month']=='Apr']['Price']
    apr_mean=np.mean(apr_price)                 #sample mean for the month of april 
    print("April mean:",apr_mean)

    loss=loss_prob(df['Chg%'])
    print("Probability of loss over stock is:",loss)

    profit=wednesday_profit(df)
    print("Probability of profit on a wednesday:",profit)
    

    plt.scatter(df['Day'], df['Chg%'])
    plt.xlabel("Day of Week")
    plt.ylabel("Chg %")
    plt.title("Chg% vs Day of Week")
    plt.show()


if __name__=="__main__":
    main()
