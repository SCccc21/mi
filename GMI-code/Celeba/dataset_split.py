import os
import numpy as np

# check, if file exists, make link
def check_link(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)  # from out_dir to in_file
        os.symlink(rel_link, link_file)
 
def add_splits(data_path):
    images_path = os.path.join(data_path, 'Img/img_align_celeba')
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
 
    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637
 
    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, test_dir)


def public_private_splits(data_path):
    file_path = os.path.join(data_path, 'identity_CelebA.txt')
    public_train_path = os.path.join(data_path,'splits_11','identity_public_train.txt')
    public_test_path = os.path.join(data_path,'splits_11','identity_public_test.txt')
    private_train_iden_path = os.path.join(data_path,'splits_11','identity_private_train.txt')
    private_test_iden_path = os.path.join(data_path,'splits_11','identity_private_test.txt')
    images_path = os.path.join(data_path, 'img_align_celeba')
    public_train_dir = os.path.join(data_path, 'splits_11', 'public', 'train')
    public_test_dir  = os.path.join(data_path, 'splits_11', 'public', 'test')
    private_train_dir = os.path.join(data_path, 'splits_11', 'private', 'train')
    private_test_dir = os.path.join(data_path, 'splits_11', 'private', 'test')

    if not os.path.exists(public_train_dir):
        os.makedirs(public_train_dir)
    if not os.path.exists(public_test_dir):
        os.makedirs(public_test_dir)
    if not os.path.exists(private_train_dir):
        os.makedirs(private_train_dir)
    if not os.path.exists(private_test_dir):
        os.makedirs(private_test_dir)

    TRAIN_STOP = int(21104 * 0.9)
    cnt_public = 0
    cnt_private = 0

    f = open(file_path, "r")
    f_public_train = open(public_train_path, "w")
    f_public_test = open(public_test_path, "w")
    f_private_train = open(private_train_iden_path, "w")
    f_private_test = open(private_test_iden_path, "w")

    for line in f.readlines():
        img_name, iden = line.strip().split(' ')
    
        if 0 < int(iden) < 1001:
            if cnt_private < TRAIN_STOP:
                check_link(images_path, img_name, private_train_dir)
                f_private_train.write(line)
            else:
                check_link(images_path, img_name, private_test_dir)
                f_private_test.write(line)
            cnt_private += 1

        if 1000 < int(iden) < 2001:
            if cnt_public < TRAIN_STOP:
                check_link(images_path, img_name, public_train_dir)
                f_public_train.write(line)
            else:
                check_link(images_path, img_name, public_test_dir)
                f_public_test.write(line)
            cnt_public += 1
            # check_link(images_path, img_name, public_dir)
            # f_public.write(line)
            

    # print('number of images in public dir is:'.format())

def combine(f, foldername, base_path):

    img_path = os.path.join(base_path, foldername)
    for i in range(1000):
        basename = "{:05d}.png".format(i+int(foldername))
        check_link(img_path, basename, base_path)
        print("creat symlink for image ", basename)
        f.write(basename)
        f.write('\n')
    
def getListOfFiles(f, dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
            f.write(entry)
            f.write('\n')
                
    return allFiles


def find(img_name):
    datafile = open('/home/sichen/data/identity_CelebA.txt', "r")
    for line in datafile.readlines():
        # import pdb; pdb.set_trace()
        if img_name in line:
            img_name, iden = line.strip().split(' ')
            return iden
    print("{} not found!".format(img_name))


def find_match():
    match = []
    datafile = open('/home/sichen/data/testset.txt', "r")
    mark = 2000000
    for line in datafile.readlines():
        img_name, label = line.strip().split(' ')
        if label == mark:
            continue
        mark = label  
        origin_iden = find(img_name)
        match.append(int(origin_iden) - 1) 
        
    print(match)
    return match

def train_test_split(base_path):
    match_list = find_match() # both key and value use 0 version

    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637
    
    train_path = os.path.join(base_path, 'train_all.txt')
    val_path = os.path.join(base_path, 'val_all.txt')
    test_path = os.path.join(base_path, 'test_all.txt')
    f_train = open(train_path, "w")
    f_val = open(val_path, "w")
    f_test = open(test_path, "w")

    for i in range(0, NUM_EXAMPLES):
        img_name = "{:06d}.png".format(i+1)
        print(img_name)
        iden = find(img_name) # iden is 1 version
        label = int(iden) - 1
        if 0 <= label <= 999:
            new_label = match_list[label]
            print("{} changed from {} to {}".format(img_name, label, new_label))
        elif label in match_list:
            new_label = match_list.index(label)
            print("{} changed from {} to {}".format(img_name, label, new_label))
        else:
            new_label = label
        
        if i in range(0, TRAIN_STOP):
            f_train.write(img_name + ' ' + str(new_label))
            f_train.write("\n")
        if i in range(TRAIN_STOP, VALID_STOP):
            f_val.write(img_name + ' ' + str(new_label))
            f_val.write("\n")
        if i in range(VALID_STOP, NUM_EXAMPLES):
            f_test.write(img_name + ' ' + str(new_label))
            f_test.write("\n")


def train_test_split_new(base_path):
    f = open('/home/sichen/data/identity_CelebA.txt', "r")
    match_list = find_match() # both key and value use 0 version
    
    train_path = os.path.join(base_path, 'train_all_new.txt')
    val_path = os.path.join(base_path, 'val_all_new.txt')
    f_train = open(train_path, "w")
    f_val = open(val_path, "w")
    cnt = np.zeros(10177)

    for line in f.readlines():
        img_name, iden = line.strip().split(' ')
        print(img_name)
        label = int(iden)-1
        if 0 <= label <= 999:
            new_label = match_list[label]
            print("{} changed from {} to {}".format(img_name, label, new_label))
        elif label in match_list:
            new_label = match_list.index(label)
            print("{} changed from {} to {}".format(img_name, label, new_label))
        else:
            new_label = label
        cnt[new_label] += 1
        if cnt[new_label] < 4:
            f_val.write(img_name + ' ' + str(new_label))
            f_val.write("\n")
        else:
            f_train.write(img_name + ' ' + str(new_label))
            f_train.write("\n")


def new_gan(base_path):
    gan_path = os.path.join(base_path, 'ganset_new.txt')
    train_path = os.path.join(base_path, 'train_pub.txt')
    test_path = os.path.join(base_path, 'test_pub.txt')
    all_file_path = os.path.join(base_path, 'identity_CelebA.txt')
    f = open(all_file_path, "r")
    f_gan = open(gan_path, "w")
    f_train = open(train_path, "w")
    f_test = open(test_path, "w")
    cnt = np.zeros(2000)  #2000 identities in total

    for line in f.readlines():
        img_name, iden = line.strip().split(' ')
        # import pdb; pdb.set_trace()
        if 1000 < int(iden) < 3001:
            label = int(iden) - 1
            name = os.path.splitext(img_name)[0]+ '.png' 
            f_gan.write(name)
            f_gan.write("\n")
            if cnt[label-1000] < 2:
                cnt[label-1000] += 1
                f_test.write(name + ' ' + str(label))
                f_test.write("\n")
            else:
                f_train.write(name + ' ' + str(label))
                f_train.write("\n")

    f_gan.close()
    f_train.close()
    f_test.close()

def mnist_split(base_path):
    public_path = os.path.join(base_path, 'ganset.txt')
    private_train = os.path.join(base_path, 'train.txt')
    private_test = os.path.join(base_path, 'test.txt')
    f_public = open(public_path, "w")
    f_private_train = open(private_train, "w")
    f_private_test = open(private_test, "w")

    # train file 60,000
    listOfFile = os.listdir(base_path)
    for entry in listOfFile:
        # import pdb; pdb.set_trace()
        # print(entry)
        if entry.endswith('.png'):
            img_name, label = os.path.splitext(entry)[0].strip().split('_')
            if int(img_name) <= 60000:
                # train
                if int(label) in [0, 1, 2, 3, 4]:
                    # private
                    f_private_train.write(entry + ' ' + str(label))
                    f_private_train.write('\n')

                else:
                    # public
                    f_public.write(entry)
                    f_public.write('\n')

            else:
                # test
                if int(label) in [0, 1, 2, 3, 4]:
                    # private
                    f_private_test.write(entry + ' ' + str(label))
                    f_private_test.write('\n')

                else:
                    # public
                    f_public.write(entry)
                    f_public.write('\n')

def facescrub(base_path):
    public_path = os.path.join('./feat', 'dist_h.txt')
    f_public = open(public_path, "w")

    listOfFile = os.listdir(base_path)
    for entry in listOfFile:
        if entry.endswith('.png'):
            f_public.write(entry)
            f_public.write('\n')



 
if __name__ == '__main__':
    # base_path = '/home/sichen/data'
    # public_private_splits(base_path)
    # base_path = '/home/sichen/data/pf83_fixed'
    # base_path = '/home/sichen/data/MNIST_imgs'
    # no = 0
    
    # for i in range(30):
    #     foldername = str(no).zfill(5)
    #     no += 1000
    #     combine(f, foldername, base_path)

    # mnist_split(base_path)

    # f.close()
    # base_path = '/home/sichen/data/facescrub/imgs'
    # base_path = '/home/sichen/mi/GMI-code/Celeba/fid/fid_dist_entropy'
    # facescrub(base_path)
    train_test_split_new('/home/sichen/data')