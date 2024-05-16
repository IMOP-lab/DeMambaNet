import os
import pickle

# from sklearn.model_selection import train_test_split

# image_dir = r'/home/lab/share/ZBF-SKIP2000/image'
# masks_path = r'/home/lab/share/ZBF-SKIP2000/mask_after_pre'
#
# images = os.listdir(image_dir)
#
# data_splits = []
# with open(r'/home/lab/share/ZBF-SKIP2000/splits.pkl', 'rb') as f:
#     data_ORI = pickle.load(f)
#
# print(len(data_ORI[0]['train']))
# print(len(data_ORI[0]['val']))
# print(len(data_ORI[0]['test']))
#
#
# for i in range(5):
#
#     train_images = data_ORI[i]['train'] + data_ORI[i]['test']
#     val_images = data_ORI[i]['val']
#     test_images = []
#     split = {
#       'train': train_images,
#       'val': val_images,
#       'test': test_images
#     }
#
#     data_splits.append(split)
#
# with open(r'/home/lab/samModel/autosam_baseline_zbf/__pycache__/splits_train3000.pkl', 'wb') as f:
#     pickle.dump(data_splits, f)
#
# with open(r'/home/lab/samModel/autosam_baseline_zbf/__pycache__/splits_train3000.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(len(data[0]['train']))
# print(len(data[0]['val']))
# print(len(data[0]['test']))
#


# # 生成test

image_dir = r'/home/zbf/Desktop/code/teech_mamba/again/image'
# image_dir = r'/home/zbf/lab/data/teech/mask'

images = os.listdir(image_dir)

data_splits = []

for i in range(5):
    train_images = []
    val_images = []
    test_images = images
    split = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    data_splits.append(split)

with open(r'/home/zbf/Desktop/code/teech_mamba/again/splits.pkl', 'wb') as f:
    pickle.dump(data_splits, f)

with open(r'/home/zbf/Desktop/code/teech_mamba/again/splits.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data[0]['train']))
print(len(data[0]['val']))
print(len(data[0]['test']))


 #train
# #
# import os
# import pickle
# from sklearn.model_selection import train_test_split
#
# image_dir = r'/home/zbf/teech/test/image'
# # masks_path = r'/home/zbf/lab/data/teech/train/mask'
#
# images = os.listdir(image_dir)
#
# data_splits = []
#
# for i in range(5):
#     test_images = []
#     train_test_images, val_images = train_test_split(images, test_size=0.099, random_state=i)
#     # train_images, test_images = train_test_split(train_test_images, test_size=0.7, random_state=i)
#     split = {
#         'train': train_test_images,
#         'val': val_images,
#         'test': test_images
#     }
#
#     data_splits.append(split)
#
# with open(r'/home/zbf/teech/test/splits.pkl', 'wb') as f:
#     pickle.dump(data_splits, f)
#
# with open(r'/home/zbf/teech/test/splits.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)
# print(len(data[0]['train']))
# print(len(data[0]['val']))
# print(len(data[0]['test']))
#

# import os
# import pickle
# from sklearn.model_selection import train_test_split
# #
# image_dir = r'/home/zbf/lab/data/teech/train/mask'
# # masks_path = r'/home/zbf/lab/data/teech/train/mask'
#
# images = os.listdir(image_dir)
#
# data_splits = []
#
# for i in range(5):
#     test_images = []
#     train_test_images, val_images = train_test_split(images, test_size=0.1, random_state=i)
#     # train_images, test_images = train_test_split(train_test_images, test_size=0.7, random_state=i)
#     split = {
#         'train': train_test_images,
#         'val': val_images,
#         'test': test_images
#     }
#
#     data_splits.append(split)
#
# with open(r'/home/zbf/lab/data/teech/train/splits.pkl', 'wb') as f:
#     pickle.dump(data_splits, f)
# #
# with open(r'/home/zbf/lab/data/teech/train/splits.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)
print(len(data[0]['train']))
print(len(data[0]['val']))
print(len(data[0]['test']))
