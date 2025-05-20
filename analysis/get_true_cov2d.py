import numpy as np
import cv2
from matplotlib import pyplot as plt


def true_conv2d(img, kernel, stride=1):
    c, h, w = img.shape
    k_c, k_h, k_w = kernel.shape

    # Calculate output dimensions
    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1

    # Initialize output array
    conv = np.zeros((out_h, out_w), dtype=np.float32)

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            img_temp = img[:, i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            conv[i, j] = np.abs(np.sum(img_temp * kernel))
    conv = (conv - conv.min()) / (conv.max() - conv.min())
    return conv


if __name__ == "__main__":
    # ---- read kernel --------------------------------------------------------
    temp=[f"../data/in/kernels/kernel_edge_{i}c.txt" for i in range(1, 4)]
    img_brg = cv2.imread(r"..\data\in\input_lena.png",
                        cv2.IMREAD_UNCHANGED)[..., :3].transpose(2, 0, 1)

    res=[]
    for kernel_file in temp:
        kernel = np.loadtxt(kernel_file, dtype=np.float32)
        kernel = kernel[1:].reshape((3, 3, 3))

        res += [true_conv2d(img_brg, kernel, stride=2)]
    res = np.array(res).transpose(1, 2, 0)

    # ---- make something you can see ----------------------------------------
    res = (res * 255).astype(np.uint8)

    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # ---- save result -------------------------------------------------------
    cv2.imwrite(r"..\data\out\true_out.png", res)
