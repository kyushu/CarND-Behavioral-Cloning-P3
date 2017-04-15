import os
import csv
import cv2
from math import fabs
import random
from shutil import copyfile, rmtree

'''
    the cvs format is 
    |      0        |     1     |     2     |     3     |     4     |     5     |     6     |     7     |     8     |
    |Center_Image   |Left_Image |Right_Image| Steering  |Throttle   | Brake     | Speed     |           |
'''

def mod_csv(main_data_dir, mode=2):

    dir_list = get_sub_data_dir(main_data_dir)

    csv_folder = main_data_dir + '/csv_file'
    rmtree(csv_folder)
    os.makedirs(csv_folder)


    for dir_path in dir_list:
        # print("dir_name: {}".format(dir_name))
        dir_name = dir_path.split('/')[-1]
        source_file = dir_path + '/' + 'driving_log.csv'
        mod_file_name = dir_name + '+' + 'driving_log.csv'
        mod_file_path = main_data_dir + '/csv_file/' + mod_file_name
        mod_file = open(mod_file_path, 'w')
        csv_writer = csv.writer(mod_file)

        with open(source_file) as fp:
            reader = csv.reader(fp)
            for line in reader:

                line[0] = dir_path +'/IMG/'+line[0].split('/')[-1]
                line[1] = dir_path +'/IMG/'+line[1].split('/')[-1]
                line[2] = dir_path +'/IMG/'+line[2].split('/')[-1]
                steer = float(line[3])

                # write 
                if mode == 1:
                    # only center image
                    csv_writer.writerow(line)
                elif mode == 2:
                    # all image indlude center, left and right
                    csv_writer.writerow(line) 
                    # add Left camera image
                    line[0] = line[1]
                    line[3] = steer + 0.2
                    csv_writer.writerow(line)
                    # add Right camera image
                    line[0] = line[2]
                    line[3] = steer - 0.2
                    csv_writer.writerow(line)
                elif mode == 3:
                    # Use thid mode will get better behavior
                    sample = random_image(line)  
                    csv_writer.writerow(sample)                  
                #     rnd = random.randint(1,3)
                #     if rnd == 1:
                #         # center
                #         csv_writer.writerow(line)
                #     elif rnd == 2:
                #         # left
                #         line[0] = line[1]
                #         line[3] = steer + 0.2
                #         csv_writer.writerow(line)
                #     elif rnd == 3:
                #         # right
                #         line[0] = line[2]
                #         line[3] = steer - 0.2
                #         csv_writer.writerow(line)
                # else:
                #     # only center image
                #     csv_writer.writerow(line)

        mod_file.close()

def random_image(sample):
    steer = float(sample[3])
    rnd = random.randint(1,3)
    if rnd == 1:
        # center
        return sample
    elif rnd == 2:
        # left
        sample[0] = sample[1]
        sample[3] = steer + 0.2
        return sample
    elif rnd == 3:
        # right
        sample[0] = sample[2]
        sample[3] = steer - 0.2
        return sample


def store_flip_image(image_path):
    path_comps = image_path.split('/')[-1]
    dir_path = path_comps[:-1]
    f_name = path_comps[-1]
    name_comps = f_name.split('.')
    f_name = name_comps[0]
    extension = name_comps[1]
    
    f_image_path = dir_path + '/' + f_name +'_flip' + '.' + extension
    
    f_image = cv2.imread(image_path)
    f_image = cv2.flip(f_image, 1)
    cv2.imwrite(f_image, f_image_path)
    return f_image_path


# get directories of data set in main data directory
def get_sub_data_dir(main_data_dir):
    dir_list = []
    for root, dirs, files in os.walk(main_data_dir, topdown=True):
        depth = root[len(os.path.sep):].count(os.path.sep)
        if depth == 1:
            # We're currently two directories in, so all subdirs have depth 3
            dir_list += [os.path.join(root, d) for d in dirs]
            dirs[:] = [] # Don't recurse any deeper
    print(dir_list)

    matching = [s for s in dir_list if "csv_file" in s]
    # print("matching:{}".format(matching))
    if len(matching) > 0:
        dir_list.remove(matching[0])

    # check csv_file directory
    csv_dir = main_data_dir + '/csv_file/'
    if os.path.isdir(csv_dir) == False:
        os.makedirs(csv_dir)

    # copy csv file to ccsv_file directory
    for dir_path in dir_list:
        source_file = dir_path + '/' + 'driving_log.csv'
        
        dir_name = dir_path.split('/')[-1]
        target_file = csv_dir + dir_name + '+' + 'driving_log.csv'
        
        copyfile(source_file, target_file)

    return dir_list



def generate_new_log(main_data_dir, target_csv_file, final_csv_file):
    # final_csv_file = '/driving_log_all.csv'
    
    
    w_filename = main_data_dir + final_csv_file
    w_file = open(w_filename, 'w')
    csv_writer = csv.writer(w_file)

    csv_dir = main_data_dir + 'csv_file/'

    file_list = [f for f in os.listdir(csv_dir) if os.path.isfile(os.path.join(csv_dir, f))]
    print("file_list:{}".format(file_list))

    matching = [s for s in file_list if "DS_Store" in s]
    if len(matching) > 0:
        file_list.remove(matching[0])

    for file_name in file_list:
        file = csv_dir + file_name
        # print("file:{}".format(file))

        # data_name = file_name.split('+')[0]
        # data_dir = main_data_dir + data_name
        # print("data_dir:{}".format(data_dir))
        
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                csv_writer.writerow(line)

    w_file.close()




if __name__ == '__main__':
    main_data_dir = 'data/'
    target_csv_file = 'driving_log.csv'
    final_csv_file = 'driving_log_all.csv'

    # sub_data_dir_list = get_sub_data_dir(main_data_dir)
    # generate_new_log(main_data_dir, target_csv_file, final_csv_file)

    mod_csv(main_data_dir)
    generate_new_log(main_data_dir, target_csv_file, final_csv_file)
