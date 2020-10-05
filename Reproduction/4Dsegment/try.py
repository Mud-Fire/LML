import pydicom

img = pydicom.read_file("F:/LML/LML/Reproduction/4Dsegment/chenyongjun/CHEN_YONG_JUN.MR.AN'ZHEN_HEART.0008.0019.2019.08.04.18.57.19.100105.34647597.IMA")

import matplotlib.pyplot as plt

plt.imshow(img.pixel_array)

# plt.show()

from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import numpy as np
import matplotlib

example_filename = './data/subject_1/example.nii'
img = nib.load(example_filename)

print(img)


print("===========================")
print(img.dataobj.shape)

width, height, queue, d = img.dataobj.shape

# OrthoSlicer3D(img.dataobj).show()

x = int((queue/10) ** 0.5) + 1
num = 1

# 按照10的步长，切片，显示2D图像
for i in range(0, width, 10):
    print(i)
    img_arr = img.dataobj[:, :, i, 1]
    img_arr = np.squeeze(img_arr)
    print(img_arr.shape)
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1

plt.show()

