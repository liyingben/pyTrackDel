# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data,io,segmentation,measure,morphology,color

#加载并裁剪硬币图片
# image = data.coins()[50:-50, 50:-50]

# thresh =filter.threshold_otsu(image) #阈值分割
# bw =morphology.closing(image > thresh, morphology.square(3)) #闭运算
#
# cleared = bw.copy()  #复制
# segmentation.clear_border(cleared)  #清除与边界相连的目标物

# label_image =measure.label(cleared)  #连通区域标记
# borders = np.logical_xor(bw, cleared) #异或
# label_image[borders] = -1
# image_label_overlay =color.label2rgb(label_image, image=image) #不同标记用不同颜色显示
#
fig,(ax0,ax1)= plt.subplots(1,2, figsize=(8, 6))
# ax0.imshow(cleared,plt.cm.gray)
# ax1.imshow(image_label_overlay)
img = io.imread('/media/liyingben/g/pix/newPath/target/2017-09-02_3110_0_0.jpg')
img = color.rgb2gray(img)
for region in measure.regionprops(img): #循环得到每一个连通区域属性集

    #忽略小区域
    if region.area < 100:
        continue

    #绘制外包矩形
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)
fig.tight_layout()