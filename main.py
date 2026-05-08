import subprocess
import pathlib
import logging
import click
import time
import numpy as np

from imageio.v3 import imread
from matplotlib import pyplot as plt


# init logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def postprocessing(img: np.ndarray) -> np.ndarray:
    """
    Apply post-processing to the image.
    Only runs this when pre-processing with simplify flag is turned on.

    :param img: the image array
    """
    img[img <= 0] = 0
    img[img >= 1] = 255
    return img


def preprocessing(img: np.ndarray, simplify: bool = True) -> np.ndarray:
    """
    Apply preprocessing to the image.
    Reverting the colors (black to white and vice versa).

    :param img: the image array
    """
    # color ranges from [0, 255]
    # simplifying to 1 will result in a faster calculation
    max_channel_no = 255
    if (simplify):
        max_channel_no = 1

    # apply preprocessing
    img[img < 128] = 0
    img[127 < img] = max_channel_no
    img = max_channel_no - img

    return img


def write_to_file(output_file: pathlib.Path, arr: np.ndarray) -> None:
    """
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


def get_ssd_map_from_file(ssd_map_file: pathlib.Path, no_of_width: int, no_of_col: int) -> np.ndarray:
    """
    Read the ssd_map from the output file.
    Parse the ssd_map from the output file to an np array.

    :param ssd_map_file: the path to the ssd_map file
    :param no_of_width: the number of width of the ssd_map array
    :param no_of_col: the number of column of the ssd_map array
    """
    if (not ssd_map_file.exists()):
        raise FileExistsError(f"'{ssd_map_file}' does not exist")

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


def generate_ssd_map(
    input_img: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Generate an SSD map.

    :param input_img: the input image in np array
    :param template: the template in np array
    """
    # get the shape
    input_img_rows = input_img.shape[0]
    input_img_cols = input_img.shape[1]

    template_rows = template.shape[0]
    template_cols = template.shape[1]

    # condition 1: input_img's row cannot be smaller than template's row
    if (input_img_rows < template_rows):
        raise ValueError("Input Image's number of rows is smaller than template's")
    
    # condition 2: input_img's column cannot be smaller than template's col
    if (input_img_cols < template_cols):
        raise ValueError("Input Image's number of rows is smaller than template's")

    # get the ssd_map shape
    ssd_map_no_of_rows = input_img_rows - template_rows + 1
    ssd_map_no_of_cols = input_img_cols - template_cols + 1

    # generate the ssd_map
    ssd_map = np.zeros((ssd_map_no_of_rows, ssd_map_no_of_cols))
    for row in range(ssd_map_no_of_rows):
        for col in range(ssd_map_no_of_cols):

            row_start = row
            row_end = row_start + template_rows

            col_start = col
            col_end = col_start + template_cols

            # create the window in the input image
            input_img_intermediate = input_img[row_start:row_end, col_start:col_end]

            # step 1: diff
            diff = template - input_img_intermediate

            # step 2: square
            square = diff ** 2
            
            # step 3: sum
            total = np.sum(square)

            # step 4: assign valueget_ssd_map to ssd_map
            ssd_map[row, col] = total

    return ssd_map


@click.command()
@click.option('--cuda', '-c', is_flag=True, help='Using CUDA code to generate SSD map')
def main(cuda: bool):

    # create 'output' folder
    output_dir_path = pathlib.Path(__file__).parent / 'output'
    if not output_dir_path.is_dir():
        output_dir_path.mkdir(exist_ok=True)

    # to speed up the generation of SSD map
    simplify = True


    # (row, col) OR (height, width) OR (108, 81)
    # load template and apply necessary preprocessing
    template = imread('./input/template.png', pilmode='L')

    # apply preprocessing on the template
    template = preprocessing(template, simplify=simplify)
    logging.info(f"Template's Shape:     {template.shape}")
    logging.info(f"Template's Unique:    {np.unique(template)}")


    # (row, col) OR (height, width) OR (1053, 745)
    # load input image and apply necessary preprocessing
    input_img = imread('./input/input.png', pilmode='L')

    # apply preprocessing on the input_img
    input_img = preprocessing(input_img, simplify=simplify)
    logging.info(f"Input Image's Shape:  {input_img.shape}")
    logging.info(f"Input Image's Unique: {np.unique(input_img)}")


    # save template to output file - for uni submission 
    template_output_file = output_dir_path / 'template'
    write_to_file(
        output_file=template_output_file,
        arr=template
    )
    logging.info(f"'template' saved to '{template_output_file}'")


    # save input image to output file - for uni submission
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


    # ---------------------------------------------
    #  Using GPU (CUDA) code
    # ---------------------------------------------
    if (cuda):
        logging.info("Using GPU to process the SSD to create the SSD Map")

        # first command: compilation - to generate a binary file
        cmd = ['nvcc', 'ssd.cu', '-o', 'ssd']
        subprocess.run(cmd)

        # second command: run the compiled file
        cmd = ['./ssd']
        subprocess.run(cmd)

        # get the ssd_map
        ssd_map_file = output_dir_path / 'ssd_map'
        ssd_map = get_ssd_map_from_file(
            ssd_map_file=ssd_map_file,
            no_of_width=number_of_width,
            no_of_col=number_of_col,
        )

    # ---------------------------------------------
    #  Using CPU
    # ---------------------------------------------
    else:
        logging.info("Using CPU to process the SSD to create the SSD Map")
        start = time.perf_counter()
        ssd_map = generate_ssd_map(
            input_img=input_img,
            template=template
        )
        end = time.perf_counter()
        logging.info(f"Time taken to generate SSD Map: {end - start:.6f} seconds")

    logging.info("SSD Map generated")

    # revert back if simplify set to True
    if (simplify):
        input_img = postprocessing(input_img)
        template = postprocessing(template)


    temp_row = template.shape[0]
    temp_col = template.shape[1]

    target_num = 5 # hardcoded
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
    plt.savefig("./output/output.png")

if (__name__ == "__main__"):
    main()