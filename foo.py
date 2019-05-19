import pandas as pd

def func(a,b,*args):
	try:
		return a+b+sum(args)
	except:
		return 0

if __name__ == "__main__":

	print(func(1,2,[1,5,3]))

	

	