import json
import cv2
import os
import shutil

def read_file(folder, set):
    path = os.path.join(folder, set+".txt")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    sample_list=[]
    for line in lines:
        words = line.split("\t")
        sample_list.append([words[0], words[1]])
    return sample_list


def save_file(folder, set, sample_list, ratio):
    set_folder=os.path.join(folder, set)
    os.makedirs(set_folder)

    annotation_str = ""
    for sample in sample_list:
        sample_path = sample[0]
        sample_file = os.path.split(sample_path)[1]
        im = cv2.imread(sample_path)
        width = int(im.shape[1] / ratio + 0.5)
        height = int(im.shape[0] / ratio + 0.5)
        resized = cv2.resize(im, (width, height), interpolation = cv2.INTER_AREA)
        # cv2.imshow("resized", resized)
        # cv2.waitKey(0)

        path = os.path.join(set_folder, sample_file)
        cv2.imwrite(path, resized)

        annotations = json.loads(sample[1])
        for annotation in annotations:
            for point in annotation["points"]:
                point[:] = [int(x / ratio+0.5) for x in point] # in place calculation

        annotation_str += sample_path + "\t" + json.dumps(annotations) + "\n"

        # for debugging purpose
        for annotation in annotations:
            points = annotation["points"]
            for i in range(len(points)-1):
                cv2.line(resized, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (0, 255, 0), 1)
            cv2.line(resized, (points[-1][0], points[-1][1]), (points[0][0], points[0][1]), (0, 255, 0), 1)
        path = path.replace(".bmp", "debug.bmp")
        cv2.imwrite(path, resized)
    
    path = os.path.join(folder, set+".txt") 
    with open(path, "w", encoding="utf-8") as f:
        lines = f.write(annotation_str)


ori_folder = "D:\\datasets\\sanqi\\det"
new_folder = "D:\\datasets\\sanqi\\det_reduced"
ratio = 2

if os.path.exists(new_folder):
    shutil.rmtree(new_folder)

set = "train"
sample_list = read_file(ori_folder, set)
save_file(new_folder, set, sample_list,  ratio)

set = "val"
sample_list = read_file(ori_folder, set)
save_file(new_folder, set, sample_list,  ratio)

set = "test"
sample_list = read_file(ori_folder, set)
save_file(new_folder, set, sample_list,  ratio)
