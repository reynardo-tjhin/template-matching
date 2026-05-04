# Importing Libraries
import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load template and apply necessary preprocessing
template = imread('./W10LabData/template.png', pilmode='L')
template = 255 -  template # reverse the grayscale color

# binarize pixels
template[template < 128] = 0
template[127 < template] = 255

print(f"Template's Shape:     {template.shape}") # (row, col) OR (height, width) OR (108, 81)
print(f"Template's Unique:    {np.unique(template)}") # print unique elements of template
# plt.imshow(template, cmap='gray')
# plt.show()

# Load input image and apply necessary preprocessing
input_img = imread('./W10LabData/input.png', pilmode='L')
input_img[input_img < 128] = 0
input_img[127 < input_img] = 255
input_img = 255 - input_img

# print(input_img)
print(f"Input Image's Shape:  {input_img.shape}")
print(f"Input Image's Unique: {np.unique(input_img)}")

# plt.imshow(input_img, cmap='gray')
# plt.show()

print("Saving template data into a file")

file = open("template", "w")
for row in range(template.shape[0]):
    for col in range(template.shape[1]):
        file.write(str(template[row, col]) + " ")
    file.write("\n")

file.close()

print("Saving input image data into a file")

file = open("input_image", "w")
for row in range(input_img.shape[0]):
    for col in range(input_img.shape[1]):
        file.write(str(input_img[row, col]) + " ")
    file.write("\n")

file.close()



number_of_width = input_img.shape[0] - template.shape[0] + 1
number_of_col = input_img.shape[1] - template.shape[1] + 1
print(f"Dimension of SSD Map: {number_of_width, number_of_col}")


print("Running another file with GPU to generate SSD Map")
print("Python main file will sleep for 10 seconds")
# Use other file to get the ssd map
import os
cmd = 'nvcc ssd.cu -o ssd'
os.system(cmd)
cmd = './ssd'
os.system(cmd)

from time import sleep
sleep(10) # sleep for 15 seconds to let the ssd.cu file generate the ssd_map


print("Generating SSD Map successful")
print("Reading from file")

# get the ssd_map
ssd_map = np.zeros((number_of_width, number_of_col))
file = open("ssd_map", "r")
data = file.readlines()
file.close()

for row in range(0, ssd_map.shape[0]):
    line = data[row].strip("\n").split(" ")
    for col in range(0, ssd_map.shape[1]):
        ssd_map[row, col] = np.float64(float(line[col]))




temp_row = template.shape[0]
temp_col = template.shape[1]

target_num = 5
target_res = []

print(f"Getting {target_num} coordinates")

i = 0
while (i < target_num):

    local_max = np.max(ssd_map)
    local_min = np.min(ssd_map)
    x, y = np.where(ssd_map == local_min)
    x, y = x[0], y[0]

    print(f"SSD value: {local_min}, coordinates: {y,x}")
    target_res.append((x, y))

    # set to max value
    for row in range(x-temp_row, x+temp_row):
        for col in range(y-temp_row, y+temp_col):
            if (row >= ssd_map.shape[0] or row < 0 or col >= ssd_map.shape[1] or col < 0):
                continue
            ssd_map[row, col] = local_max

    i += 1



print("Preparing for the final image")

# get the final image
final_img = np.zeros((input_img.shape[0], input_img.shape[1]))

# assign white color to the highlighted ones in target result
for (x, y) in target_res:
    for row in range(0, template.shape[0]):
        for col in range(0, template.shape[1]):
            final_img[x+row, y+col] = 255


separator = np.multiply(np.ones((input_img.shape[0], 30)), 255)
combined_img = np.concatenate((input_img, separator), 1)
combined_img = np.concatenate((combined_img, final_img), 1)
plt.imshow(combined_img, cmap='gray')
plt.title("Name: Reynardo Tjhin   SID: 500631832")
plt.show()


