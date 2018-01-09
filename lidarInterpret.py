import numpy as np
# difference in 50 of timestamps is half a millisecond
def interpretCSV(path):
	final = []
	recent = 0
	add = []
	limit = 100
	for line in open(path,"r"):
		line = line.strip("\n")
		line = line.split(",")
		if line[0] != "-1":
			if line[0] == "TIMESTAMP":
				continue
			if (recent == 0):
				recent = float(line[0])
			elif (recent + 50.0) < float(line[0]):
				print(np.shape(add))
				if len(add) < limit:
					print("Error, ADD Under Limit")
				if len(add) > limit:
					print("add too big")
					x = len(add) - limit
					add = add[x:]
					print("New add",add)
				final += [add]
				add = []
		add += [line[1:]]
	print(final)

	print("Final shape",np.shape(final))
	final = map(int,final)
	print("final",np.shape(final))


interpretCSV("./lidartest.csv")
