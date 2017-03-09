import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="image_size")
    parser.add_argument("width")
    parser.add_argument("height")
    return parser.parse_args()


arguments = get_arguments()

with open("groundtruth_rect.txt", "r") as f:
    with open("features_groundtruth.txt", "w") as f2:
        doc = f.read().splitlines()
        for line in doc:
            values = line.split(",")
            values = list(map(int, values))
            values[0] = int((values[0] * 147)/int(arguments.width))
            values[1] = int((values[1] * 147) / int(arguments.height))
            values[2] = int((values[2] * 147) / int(arguments.width))
            values[3] = int((values[3] * 147) / int(arguments.height))
            f2.write(str(values[0])+","+str(values[1])+","+str(values[2])+","+str(values[3])+"\n")



