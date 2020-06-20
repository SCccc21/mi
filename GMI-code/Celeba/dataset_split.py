import os

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
    

 
if __name__ == '__main__':
    base_path = '/home/sichen/data'
    public_private_splits(base_path)
    # train_test_split('/home/sichen/data/splits_11/private')