with open("features_groundtruth.txt", "r") as f:
    with open("groundtruth.txt", "w") as f2:
        doc = f.read().splitlines()
        for line in doc:
            values = line.split(",")
            values = list(map(int, values))
            x1 = values[0]
            y1 = values[1]
            x2 = x1
            y2 = y1 + values[3]
            x3 = x1 + values[2]
            y3 = y2
            x4 = x3
            y4 = y1
            f2.write(str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(x3)+","+str(y3)+","+str(x4)+","+str(y4)+"\n")



