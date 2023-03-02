from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
ascent_image = misc.ascent()
image_transformed = np.copy(ascent_image)
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]
# filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

weight = 1
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (ascent_image[x-1, y-1] * filter[0][0])
        convolution = convolution + (ascent_image[x-1, y] * filter[0][1])
        convolution = convolution + (ascent_image[x-1, y+1] * filter[0][2])
        convolution = convolution + (ascent_image[x, y-1] * filter[1][0])
        convolution = convolution + (ascent_image[x, y] * filter[1][1])
        convolution = convolution + (ascent_image[x, y+1] * filter[1][2])
        convolution = convolution + (ascent_image[x+1, y-1] * filter[2][0])
        convolution = convolution + (ascent_image[x+1, y] * filter[2][1])
        convolution = convolution + (ascent_image[x+1, y+1] * filter[2][2])

        convolution = convolution * weight

        if(convolution<0):
            convolution = 0
        elif(convolution>255):
            convolution = 255

        image_transformed[x, y] = convolution



new_x = int(size_x/2)
new_y = int(size_y/2)
new_image = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels=[]
        pixels.append(image_transformed[x, y])
        pixels.append(image_transformed[x+1, y])
        pixels.append(image_transformed[x, y+1])
        pixels.append(image_transformed[x+1, y+1])

        new_image[int(x/2), int(y/2)] = max(pixels)


plt.grid(False)
plt.gray()
plt.imshow(new_image)
plt.show()
