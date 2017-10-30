import pickle

with open('f1.voc.pkl','rb') as file:
	dic = pickle.load(file)
	print(dic)
