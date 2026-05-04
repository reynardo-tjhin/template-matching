#include <stdio.h>

#define cudaCheckErrors(msg) \
do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
} while (0)



// kernel function
__global__ void ssd(int *temp_img, int *inpt_img, double *res, \
                                        int res_row, int res_col, \
                                        int temp_row, int temp_col, \
                                        int input_row, int input_col) {

    // get the threadId
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // (946, 665)
    int start_pos_x = threadId / res_col;
    int start_pos_y = threadId % res_col;

    // calculate the ssd
    double sum = 0;
    for (int i = 0; i < 108; i++) {
        for (int j = 0; j < 81; j++) {
            int inpt_img_index = start_pos_x * input_col + start_pos_y;
            int temp_img_index = i * temp_col + j;

            int inpt_img_pixel = inpt_img[inpt_img_index];
            int temp_img_pixel = temp_img[temp_img_index];

            // calculate
            double ssd = pow((temp_img_pixel - inpt_img_pixel), 2);
            sum += ssd;

            start_pos_y += 1;
        }
        start_pos_y = threadId % res_col;
        start_pos_x += 1;
    }

    // store
    res[threadId] = sum;
}

// main function
int main() {

    clock_t begin = clock();

    int template_row = 108;
    int template_col = 81;

    int input_img_row = 1053;
    int input_img_col = 745;

    int template_n = template_row * template_col;
    int input_img_n = input_img_row * input_img_col;

    // (HOST) template and image vector
    int *h_temp_img;
    int *h_inpt_img;
    h_temp_img = (int *) malloc(template_n * sizeof(int));
    h_inpt_img = (int *) malloc(input_img_n * sizeof(int));

    int result_row = input_img_row - template_row + 1;
    int result_col = input_img_col - template_col + 1;

    // (HOST) result storing the ssd
    int result_n = result_row * result_col;
    double *h_res;
    h_res = (double *) malloc(result_n * sizeof(double));

    // get template data from the file
    FILE* fpointer = fopen("output/template", "r");
    int number; int i = 0;
    while ( fscanf(fpointer, "%d", &number) == 1 ) { 
        h_temp_img[i] = number;
        i++;
    }
    fclose(fpointer);

    // get input image data from the file
    fpointer = fopen("output/input_image", "r");
    i = 0;
    while ( fscanf(fpointer, "%d", &number) == 1 ) { 
        h_inpt_img[i] = number;
        i++;
    }
    fclose(fpointer);

    printf("Finish Extracting Images Data\n");

    // (DEVICE) template and image vector
    int *d_temp_img;
    int *d_inpt_img;
    cudaMalloc(&d_temp_img, template_n * sizeof(int));
    cudaMalloc(&d_inpt_img, input_img_n * sizeof(int));

    // (DEVICE) result storing the ssd
    double *d_res;
    cudaMalloc(&d_res, result_n * sizeof(double));

    cudaCheckErrors("cudaMalloc Fails!\n");

    // Copy data from host to device
    cudaMemcpy(d_temp_img, h_temp_img, template_n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inpt_img, h_inpt_img, input_img_n * sizeof(int), cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy Fails!\n");

    printf("Running the kernel function...\n");

    // make sure that result_col is not greater than 1024
    ssd<<<result_row, result_col>>>(d_temp_img, d_inpt_img, d_res, \
                                                result_row, result_col, \
                                                template_row, template_col, \
                                                input_img_row, input_img_col);

    cudaCheckErrors("running kernel function Fails!\n");

    // Copy result from device to host
    cudaMemcpy(h_res, d_res, result_n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaCheckErrors("cudaMemcpy Fails!\n");

    // Free
    cudaFree(d_temp_img);
    cudaFree(d_inpt_img);
    cudaFree(d_res);

    cudaCheckErrors("cudaFree Fails!\n");

    // cudasynchro

    // write result to file
    fpointer = fopen("output/ssd_map", "w");
    for (int i = 0; i < result_row; i++) {
        for (int j = 0; j < result_col; j++) {
            fprintf(fpointer, "%.1f ", h_res[i * result_col + j]);
        }
        fprintf(fpointer, "\n");
    }
    fclose(fpointer);

    free(h_temp_img);
    free(h_inpt_img);
    free(h_res);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Finish execution in %f  s\n", time_spent);

    return 0;
}