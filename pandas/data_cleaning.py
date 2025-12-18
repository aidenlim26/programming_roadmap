import pandas as pd
#Data cleaning = the process of fixing/removing:
#                               1. incomplete, incorrect, or irrelevant data
#                               2. ~75% of work done w/ pandas is data cleaning

df = pd.read_csv("IRIS_dataset.csv")

# 1. Drop irrelevant columns
#df = df.drop(columns=["petal_length","petal_width"])    #make it df = ... to reassign it and make sure changes are saved

# 2. Handle missing data
#df = df.dropna(subset=["Type2"])   #dropna is "drop not available"
                                    #inside subset[], u add the name of the column that has missing data
                                    #if theres missing data, the entire row would be deleted
#df = df.fillna({"Type2":"None"})   #fillna is "fill not available"
                                    #inside the function u add a dictionary
                                    #inside the dictionary the key is the column name
                                    #and the value is the string u replace the missing data with

# 3. Fix inconsistent values
df["species"] = df["species"].replace({"Iris-setosa":"SETOSA"})         #df[], select column name inside brackets
                                                                        #replace function
                                                                        #inside function add a dictionary
                                                                        #dictionary, key=current name
                                                                        #value=what u want to replace it with
                                                                        #you can add multiple things to change in the dictionary

# 4. Standardise text
df["species"] = df["species"].str.lower()        #Make all values in the column lower case

# 5. Fix data types
#df["Legendary"] = df["Legendary"].astype(bool)  #astype function changes the datatype

# 6. Remove duplicate values
df = df.drop_duplicates()           #Deletes all duplicate rows

print(df.to_string())