import numpy as np
import cv2
import os, glob

K=3
IMGS_ROOT = "images"
TARGET_SIZE = (64, 64)

def show_img(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_ref_imgs():
    imgs_path = os.path.join(IMGS_ROOT, "reference")
    cat_imgs_path = os.path.join(imgs_path, "cats")
    dog_imgs_path = os.path.join(imgs_path, "dogs")
    imgs_with_tag = list()

    for f in glob.glob(os.path.join(cat_imgs_path, "*")):
        img = cv2.imread(f)
        img = cv2.resize(img, TARGET_SIZE)
        imgs_with_tag.append((img, "cat"))
    for f in glob.glob(os.path.join(dog_imgs_path, "*")):
        img = cv2.imread(f)
        img = cv2.resize(img, TARGET_SIZE)
        imgs_with_tag.append((img, "dog"))
    return imgs_with_tag

def process(img):
    imgs_with_tag = load_ref_imgs()
    diff_result = list()

    for tag_img, tag in imgs_with_tag:
        diff_vec = np.absolute(img - tag_img) / 255.
        diff = np.mean(diff_vec)
        diff_result.append((diff, tag))

    cat_cnt = 0
    dog_cnt = 0

    diff_result.sort(key=lambda x:x[0])
    for i in range(K):
        _, tag = diff_result[i]
        if tag == "cat":
            cat_cnt += 1
        elif tag == "dog":
            dog_cnt += 1

    if cat_cnt > dog_cnt:
        print("It is cat")
    elif cat_cnt < dog_cnt:
        print("It is dog")
    else:
        print("I have no idea.")

def main():
    select = None
    while True:
        print("Please enter a number (1~20).")
        val = input()
        if not val.isnumeric():
            print("Not a nmuber or not integer.")
            continue
        select = int(val)
        if select < 1 or select > 20:
            print("Out or the range.")
            continue
        break
    imgs_path = os.path.join(IMGS_ROOT, "test")
    img = cv2.imread(os.path.join(imgs_path, "pic{}.jpg".format(select)))
    img = cv2.resize(img, TARGET_SIZE)

    process(img)

if __name__ == '__main__':
    main()
