import matplotlib.pyplot as plt


def imgshow (img,imgName="img"):
    plt.imshow(img, cmap="gray")
    plt.title(f"{imgName}")
    plt.axis("off")
    plt.show()