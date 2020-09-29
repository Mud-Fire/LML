import pickle

with open('../data/inputdata_DL.pkl','rb') as f:c3 = pickle.load(f)
x_full = c3[0]
y_full = c3[1]
print("===============================================================")
print(len(x_full[0]))
print(x_full[0])
print(y_full[0:10])

del c3

with open('../data/inputdata_conv.pkl','rb') as f:conv = pickle.load(f)

x_full = conv[0]
y_full = conv[1]
print("===============================================================")
print(len(x_full[:,0]))
print(x_full[0])
print(y_full[0:10])
del conv