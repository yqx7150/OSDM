import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

img = nib.load('segmentation-0.nii.gz')
print(img.shape)  # shape(240, 240, 155)
print(img.header['db_name'])
width, height, queue = img.dataobj.shape  # 由文件本身维度确定，可能是3维，也可能是4维
# print("width",width)  # 240
# print("height",height) # 240
# print("queue",queue)   # 155
nib.viewers.OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()
————————————————
版权声明：本文为CSDN博主「JWangwen」的原创文章，遵循CC
4.0
BY - SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / qq_42740834 / article / details / 124473611