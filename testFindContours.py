# coding=utf-8

import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure,color,io,transform

path = '/tmp/2017-10-01_3373_0_0.png'
img = Image.open(path)#打开图片
newImg = Image.new("RGBA",(260,260),(255,255,255))
newImg.paste(img,(2,2))
newImg.save(path)
img = io.imread(path)

img = color.rgb2gray(img)
#检测所有图形的轮廓
contours = measure.find_contours(img, 0.8)


#绘制轮廓
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
ax0.imshow(img, plt.cm.gray)
ax1.imshow(img, plt.cm.gray)
for n, contour in enumerate(contours):
    # contour =measure.approximate_polygon(contour,  tolerance=2)
    # contour =measure.subdivide_polygon(contour, degree=4)
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    print contour
ax1.axis('image')
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()



