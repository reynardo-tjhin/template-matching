import subprocess
import pathlib
import logging
import numpy as np

from imageio.v3 import imread
from matplotlib import pyplot as plt


# init logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_img(img_path: str, pilmode: str = 'L') -> np.ndarray:
    """
    Load an image based on the image path using `imread`.
    Convert the image to grayscale (pilmode = 'L').
    Apply preprocessing:
    - values less than 128: 0
    - values greater than 127: 255
    
    :param img_path: the path to the image
    :param pilmode: the PIL mode ('L' for grayscale, 'RGB', 'RGBA')
    """
    # load input image and convert the image to grayscale
    img = imread(img_path, pilmode=pilmode)

    # apply preprocessing
    img[img < 128] = 0
    img[127 < img] = 255
    img = 255 - img

    return img


def write_to_file(output_file: pathlib.Path, arr: np.ndarray) -> None:
    """    
    # ssd_map = np.zeros((number_of_width, number_of_col))
    # file = open("output/ssd_map", "r")
    # data = file.readlines()
    # file.close()

    # for row in range(0, ssd_map.shape[0]):
    #     line = data[row].strip("\n").split(" ")
    #     for col in range(0, ssd_map.shape[1]):
    #         ssd_map[row, col] = np.float64(float(line[col]))
    Write the array (numpy 2d-array) to the output file in the repository directory

    :param output_file: the output file to write to
    :param template: the input to be written to the file
    """
    # output the template to the file
    with output_file.open("w") as fp:
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                fp.write(str(arr[row, col]) + " ")
            fp.write("\n")


def get_ssd_map(ssd_map_file: pathlib.Path, no_of_width: int, no_of_col: int) -> np.ndarray:
    """
    Read the ssd_map from the output file.
    Parse the ssd_map from the output file to an np array.

    :param ssd_map_file: the path to the ssd_map file
    :param no_of_width: the number of width of the ssd_map array
    :param no_of_col: the number of column of the ssd_map array
    """
    # initialise the ssd_map np array
    ssd_map = np.zeros((no_of_width, no_of_col))

    # read from the file ran from nvcc
    data = []
    with ssd_map_file.open("r") as fp:
        data = fp.readlines()

    # convert the data to numpy array
    for row in range(0, ssd_map.shape[0]):
        line = data[row].strip("\n").split(" ")
        for col in range(0, ssd_map.shape[1]):
            ssd_map[row, col] = np.float64(float(line[col]))

    return ssd_map


def main():

    # create 'output' folder
    output_dir_path = pathlib.Path(__file__).parent / 'output'
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(exist_ok=True)


    # load template and apply necessary preprocessing
    template = load_img('./W10LabData/template.png')

    # (row, col) OR (height, width) OR (108, 81)
    logging.info(f"Template's Shape:     {template.shape}")
    logging.info(f"Template's Unique:    {np.unique(template)}")


    # load input image and apply necessary preprocessing
    input_img = load_img('./W10LabData/input.png')

    logging.info(f"Input Image's Shape:  {input_img.shape}")
    logging.info(f"Input Image's Unique: {np.unique(input_img)}")


    # save template to output file    
    template_output_file = output_dir_path / 'template'
    write_to_file(
        output_file=template_output_file,
        arr=template
    )
    logging.info(f"'template' saved to '{template_output_file}'")


    # save input image to output file
    input_img_output_file = output_dir_path / 'input_image'
    write_to_file(
        output_file=input_img_output_file,
        arr=input_img
    )
    logging.info(f"'input_img' saved to '{input_img_output_file}'")


    # ssd refers to "sum of squared differences"
    number_of_width = input_img.shape[0] - template.shape[0] + 1
    number_of_col = input_img.shape[1] - template.shape[1] + 1
    logging.info(f"Dimension of SSD Map: {number_of_width, number_of_col}")


    logging.info("Using GPU to process the SSD to create the SSD Map")

    # first command
    cmd = ['nvcc', 'ssd.cu', '-o', 'ssd']
    subprocess.run(cmd)

    # second command
    cmd = ['./ssd']
    subprocess.run(cmd)

    logging.info("SSD Map generated")


    # get the ssd_map
    ssd_map_file = output_dir_path / 'ssd_map'
    ssd_map = get_ssd_map(
        ssd_map_file=ssd_map_file,
        no_of_width=number_of_width,
        no_of_col=number_of_col,
    )
    logging.info("SSD Map loaded successfully") 


    temp_row = template.shape[0]
    temp_col = template.shape[1]

    target_num = 5
    target_res = []

    logging.info(f"Getting {target_num} coordinates")

    i = 0
    while (i < target_num):

        local_max = np.max(ssd_map)
        local_min = np.min(ssd_map)
        x, y = np.where(ssd_map == local_min)
        x, y = x[0], y[0]

        logging.info(f"SSD value: {local_min}, coordinates: {y,x}")
        target_res.append((x, y))

        # set to max value
        for row in range(x-temp_row, x+temp_row):
            for col in range(y-temp_row, y+temp_col):
                if (row >= ssd_map.shape[0] or row < 0 or col >= ssd_map.shape[1] or col < 0):
                    continue
                ssd_map[row, col] = local_max

        i += 1



    logging.info("Preparing for the final image")

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

if (__name__ == "__main__"):
    main()