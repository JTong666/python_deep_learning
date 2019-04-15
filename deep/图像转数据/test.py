from PIL import Image
import os.path
import glob
import numpy as np
def convertpng(pngfile, width=56, height=56):
    im = Image.open(pngfile)
    new_im = im.resize((width, height), Image.BILINEAR)
    new_im = new_im.convert("L")
    data = new_im.getdata()
    data = np.matrix(data)
    print(data.shape)
    return data
def main():
    last_data = np.zeros((40, 3136))
    labels = np.zeros((40, 4))
    i = 0
    for pngfile in glob.glob("D:/python_text/图像转数据/状态1/*.png"):
        a = convertpng(pngfile)
        last_data[i] = a[0]
        i += 1
    for pngfile in glob.glob("D:/python_text/图像转数据/状态2/*.png"):
        a = convertpng(pngfile)
        last_data[i] = a[0]
        i += 1
    for pngfile in glob.glob("D:/python_text/图像转数据/状态3/*.png"):
        a = convertpng(pngfile)
        last_data[i] = a[0]
        i += 1
    for pngfile in glob.glob("D:/python_text/图像转数据/状态4/*.png"):
        a = convertpng(pngfile)
        last_data[i] = a[0]
        i += 1
    print(last_data)
    np.savetxt("data.txt", last_data)
    m = np.loadtxt("data.txt")
    print(m)
    for i in range(10):
        labels[i, 0] = 1
    for i in range(10):
        labels[i+10, 1] = 1
    for i in range(10):
        labels[i+20, 2] = 1
    for i in range(10):
        labels[i+30, 3] = 1
    print(labels)
    np.savetxt("labels.txt", labels)
    print(m.shape)
    print(labels.shape)



if __name__=='__main__':
    main()



