#Machine Learning - Lab Assignment 1
#Viyaneeta Ramesh - BL.SC.U4AIE24054
#Count of number of vowels and consonants

def count_vowels_consonants(str1):
    #intialize count 0
    vowel_count=0
    consonant_count=0
    vowels="aeiou"
    for i in str1:
        if i.isalpha(): #to ignore special symbols, white spaces, punctuations
            if i in vowels:
                vowel_count+=1
            else:
                consonant_count+=1
    return vowel_count, consonant_count


s=input("Enter a string: ")
str1=s.lower() #to prevent case sensitive behaviour
x,y=count_vowels_consonants(str1) #function call

print("Vowel count is:" , x)
print("Consonant count is:" , y)
    

