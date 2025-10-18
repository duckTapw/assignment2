#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>


bool DEBUG = false;

void conv2d(
float **f, // input feature map
int H, // input height,
int W, // input width
float **g, // input kernel
int kH, // kernel height
int kW, // kernel width
float **output
)
{
    printf(":::Starting Convolution:::\n");
    struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);

    // Output array will be the same size as the feature map
    float *retval = malloc(H * W * sizeof(float));

    float sum = 0;
    float *fm = *f;
    float *km = *g;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int j = 0; j < H; j++)
    {
        for (int i = 0; i < W; i++)
        {
            #pragma omp parallel for collapse(2) reduction(+:sum) 
            for (int jj = 0; jj < kH; jj++)
            {
                for (int ii = 0; ii < kW; ii++)
                {
                    
                    int hoffset = (((int)floor(kH/2)) - jj);
                    int woffset = (((int)floor(kW/2)) - ii);
                    // printf("height offset: %i\n", i - woffset);
                    // printf("width offset: %i\n", j - hoffset);
                    if ((i - woffset) >= 0 && (j - hoffset) >= 0 && (i - woffset) < W && (j - hoffset) < H)
                    {
                        sum += km[kW * jj + ii] * fm[W * (j - hoffset) + i - woffset];
                        // printf("adding %f\n", sum);
                    }
                    else
                    {
                        // printf("adding 0.0\n");
                    }
                    
                }
            }
            retval[W * j + i] = sum;
            // printf("setting [%i][%i] as %f\n", j, i, sum);   
            sum = 0; 
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
	double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) /1e9;
    
    *output = retval;
    printf("::::::::::::::::::::::::::\n");
    printf(":::Convolution Finished:::\n");
	printf("Time Elapsed: %f seconds\n", elapsed);
}

void getmatrix(char **file, int *width, int *height, float** retval)
{
    FILE *fp;
    fp = fopen(*file, "r");
    if (fp == NULL)
    {
        printf("MISSING FILE: %s\n", *file);
        return;
    }

    int h = 0;
    int w = 0;
    //Get the width and height from the first two values
    if(fscanf(fp, "%d %d", &h, &w) != 2)
        return;
    if(DEBUG)
        printf("%dx%d\n", h, w);

    //Check these describe a real array
    if (h < 1 || w < 1)
        return;
    float *m = malloc(sizeof(float)* h * w);
    for(int j = 0; j < h; j++)
    {
        for(int i = 0; i < w; i++)
        {
            if(fscanf(fp, "%f", &m[w * j + i]) != 1)
            {
                return;
            }
            else if (DEBUG)
            {
                printf("%10.3f", m[w * j + i]);
            }
        }
        if (DEBUG)
            printf("\n");
    }
    fclose(fp);

    *width = w;
    *height = h;
    *retval = m;
}


void randMatrix(int h, int w, float **matrix)
{
    struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
    float *m = malloc(h * sizeof(float) * w);
    printf("Starting Randomisation: H=%i, W=%i\n", h, w);
    float a = 10.0;
    #pragma omp parallel for collapse(2) firstprivate(a)
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            m[w * i + j] = ((float)rand()/(float)(RAND_MAX));// * a;
            //printf("adding value: %10.3f", matrix[i][j]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
	double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) /1e9;
	printf("---------------------------------------\n");
    printf("Generated at:   %p\n", &m);
    printf("In Time:    %f seconds\n", elapsed);
    printf("Using memory:   %ld\n", h * sizeof(float) * w);
    printf("---------------------------------------\n");
    *matrix = m;
}

void storeMatrix(char* file, int h, int w, float**matrix)
{
    float *output = *matrix;
    FILE *fp = fopen(file, "w");
    if (fp != NULL)
    {
        fprintf(fp, "%i %i\n", h, w);
        for(int j = 0; j < h; j++)
        {
            for(int i = 0; i < w; i++)
            {
                fprintf(fp, "%0.3f ", output[w * j + i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}

/*
*   TODO:   
*   - Read 2 files into float32 arrays 
*   - Parallel Convolution function to calculate output
*   - Generate Random input with size H x W, and kernel with size kH x kW
*       - Write output to a file
*   - Use "getopt"????
*   - Timer for convolution
*/
int main(int argc, char **argv) 
{
    //Flags//
    // Featuremap filename/path
    char* featuremap = NULL;
    // kernel filename/path
    char* kernel = NULL;
    // Output filename/path
    char* outputfile = NULL;
    // Dimensions for matrices
    int height = 0;
    int width = 0;
    int kheight = 0;
    int kwidth = 0;
    
	omp_set_num_threads(4);
    
    //Handle arguements
    for (int i = 1; i < argc; i++)
    {
        char f = argv[i][1]; //flags start with a '-'
        switch(f)
        {
            case 'f':
                i++;
                featuremap = malloc(sizeof(argv[i]));
                featuremap = argv[i];
                break;
            case 'g':
                i++;
                kernel = malloc(sizeof(argv[i]));
                kernel = argv[i];
                break;
            case 'o':
                i++;
                outputfile = malloc(sizeof(argv[i]));
                outputfile = argv[i];
                break;
            case 'H':
                i++;
                height = atoi(argv[i]); 
                break;
            case 'W':
                i++;
                width = atoi(argv[i]); 
                break;
            case 'k':
                //if it's a k, grab the next char....
                if (argv[i][2] == 'H')
                {
                    i++;
                    kheight = atoi(argv[i]); 
                }
                else if (argv[i][2] == 'W')
                {
                    i++;
                    kwidth = atoi(argv[i]); 
                }
                else
                {
                    printf("incorrect k arg");
                }
                break;
            case 'D':
                //For printing extra statements
                DEBUG = true;
                break;
            default:
                //No need to quit if we have an invalid flag
                //Just ignore it and we'll move on to validation
                printf("Ignoring invalid flag: %s\n", argv[i]);
        }
    }


    //Define the matrices and their dimensions
    float* fmatrix = NULL;
    float* kmatrix = NULL;
    float* output = NULL;


    //Check we have all requirements
    if (height != 0 && kheight != 0)
    {
        //printf("first matrix\n");
        randMatrix(height, width, &fmatrix);
        printf("Recieved at: %p\n", &fmatrix);
        //printf("second matrix\n");
        randMatrix(kheight, kheight, &kmatrix);
        printf("Recieved at:    %p\n", &kmatrix);
        if(DEBUG)
        {
            for(int j = 0; j < height; j++)
            {
                for(int i = 0; i < width; i++)
                {
                    printf("%10.3f", fmatrix[width * j + i]);
                }
                printf("\n");
            }
        }
        if(kernel != NULL)
        {
            storeMatrix(kernel, kheight, kwidth, &kmatrix);
        }
        if(featuremap != NULL)
        {
            storeMatrix(featuremap, height, width, &fmatrix);
        }

        conv2d(&fmatrix, height, width, &kmatrix, kheight, kwidth, &output);

    }
    else if (kernel != NULL || featuremap != NULL)
    {
        // DEBUG flag checks! //////////////////////
        if(DEBUG)
        {
            printf("Feature Map file: %s\nkernel file: %s\n", featuremap, kernel);
            if (outputfile != NULL)
            {
                printf("Output file: %s\n", outputfile);
            }
        }
        ////////////////////////////////////////////

        getmatrix(&featuremap, &width, &height, &fmatrix);
        //printf("out\n");
        printf("Recieved at: %p\n", &fmatrix);

        getmatrix(&kernel, &kwidth, &kheight, &kmatrix);
        printf("Recieved at: %p\n", &kmatrix);
        if (DEBUG)
        {
            for(int j = 0; j < height; j++)
            {
                for(int i = 0; i < width; i++)
                {
                    //printf("j: %i, i: %i ", j, i);
                    printf("%10.3f", fmatrix[width * j + i]);
                }
                printf("\n");
            }
            for(int j = 0; j < kheight; j++)
            {
                for(int i = 0; i < kwidth; i++)
                {
                    //printf("j: %i, i: %i ", j, i);
                    printf("%10.3f", kmatrix[width * j + i]);
                }
                printf("\n");
            }
        }

        conv2d(&fmatrix, height, width, &kmatrix, kheight, kwidth, &output);
    }
    else
    {
        printf("Missing matrix!, -g or -f option\n");
        return 1;
    }



    if (outputfile != NULL)
    {
        storeMatrix(outputfile, height, width, &output);
    }
    
    free(fmatrix);
    free(kmatrix);
    free(output);

    return 0;
}



