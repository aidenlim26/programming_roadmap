from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)      #splits the data into X and y

#print(X)
#print(y)

df = load_breast_cancer(as_frame=True).frame.to_string()    #makes the data into a dataframe
print(df)