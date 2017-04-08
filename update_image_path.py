import os
import csv
import cv2

data_dir = 'data/'
dir_list = []
for root, dirs, files in os.walk(data_dir, topdown=True):
    depth = root[len(os.path.sep):].count(os.path.sep)
    if depth == 1:
        # We're currently two directories in, so all subdirs have depth 3
        dir_list += [os.path.join(root, d) for d in dirs]
        dirs[:] = [] # Don't recurse any deeper
print(dir_list)


w_filename = data_dir + '/driving_log_all.csv'
w_file = open(w_filename, 'w')
csv_writer = csv.writer(w_file)

for dir in dir_list:
    data_path = dir + '/driving_log.csv'
    
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            c_file = line[0].split('/')[-1]
            line[0] = dir +'/IMG/'+line[0].split('/')[-1]
            line[1] = dir +'/IMG/'+line[1].split('/')[-1]
            line[2] = dir +'/IMG/'+line[2].split('/')[-1]

            csv_writer.writerow(line)


w_file.close()
# test = ['data/curve1_0804_2017/IMG/center_2017_04_08_20_46_57_213.jpg',
# 'data/curve1_0804_2017/IMG/left_2017_04_08_20_46_57_213.jpg',
# 'data/curve1_0804_2017/IMG/right_2017_04_08_20_46_57_213.jpg']

# for img in test:
#     image = cv2.imread(img)
#     cv2.imshow("test", image)
#     cv2.waitKey(0)
