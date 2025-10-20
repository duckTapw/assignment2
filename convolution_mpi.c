#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <mpi.h>


bool DEBUG = false;

void conv2d(
float **f, // input feature map
int H, // input height,
int W, // input width
float **g, // input kernel
int kH, // kernel height
int kW, // kernel width
int sH, //stride height
int sW, //stride width
float **output
)
{
    
    
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
    //initalize the array
    struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
    float *m = malloc(h * sizeof(float) * w);
    printf("Starting Randomisation: H=%i, W=%i\n", h, w);
    float a = 10.0;
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
*   - Generate Random input with size H x W, and kernal with size kH x kW
*       - Write output to a file
*   - Use "getopt"????
*   - Timer for convolution
*/
int main(int argc, char **argv) 
{
    //Flags//
    // Featuremap filename/path
    char* featuremap = NULL;
    // Kernal filename/path
    char* kernal = NULL;
    // Output filename/path
    char* outputfile = NULL;
    // Dimensions for matrices
    int height = 0;
    int width = 0;
    int kheight = 0;
    int kwidth = 0;
    int sheight = 1;
    int swidth = 1;

    //Define the matrices and their dimensions
    float* fmatrix = NULL;
    float* kmatrix = NULL;

    int rank, size;
    MPI_Status status;
    printf("strating mpi\n");
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("in mpi\n");
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
                kernal = malloc(sizeof(argv[i]));
                kernal = argv[i];
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
            case 's':
                if(argv[i][2] == 'H')
                {
                    i++;
                    sheight = atoi(argv[i]);
                }
                else if (argv[i][2] == 'W')
                {
                    i++;
                    swidth = atoi(argv[i]);
                }
                else
                {
                    printf("incorrect s arg");
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
    
    if (rank == 0)
    {
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
            if(kernal != NULL)
            {
                storeMatrix(kernal, kheight, kwidth, &kmatrix);
            }
            if(featuremap != NULL)
            {
                storeMatrix(featuremap, height, width, &fmatrix);
            }

        }
        else if (kernal != NULL || featuremap != NULL)
        {
            // DEBUG flag checks! //////////////////////
            if(DEBUG)
            {
                printf("Feature Map file: %s\nKernal file: %s\n", featuremap, kernal);
                if (outputfile != NULL)
                {
                    printf("Output file: %s\n", outputfile);
                }
            }
            ////////////////////////////////////////////

            getmatrix(&featuremap, &width, &height, &fmatrix);
            //printf("out\n");
            printf("Recieved at: %p\n", &fmatrix);

            getmatrix(&kernal, &kwidth, &kheight, &kmatrix);
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

        }
        else
        {
            printf("Missing matrix!, -g or -f option\n");
            return 1;
        }

        //add padding to feature map.
        storeMatrix("fmap.txt", height, width, &fmatrix);
        storeMatrix("kmap.txt", kheight, kwidth, &kmatrix);
    }
    // int pH = (kheight - 1)/2;
    // int pW = (kwidth - 1)/2;
    // // //calculate padding for same size
    // // int pH = (kH - 1)/2;
    // // int pW = (kW - 1)/2;
    // // //calculate output size
    // // int oH = (H - kH + pH + sH)/sH;
    // // int oW = (W - kW + pW + sW)/sW;
    // // //set output array size
    // // float *retval = malloc(oH * oW * sizeof(float));
    // int rank, size, position = 0;

    // char buffer[kheight*width];

    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);

    // if (rank = 0)
    // {
    //     // Root process
    //     // here we'll look to divide up and send out the work
    //     // before piecing it back together.
    //     // We'll also start the timer here.
    //     printf(":::Starting Root Matrix Splitting:::\n");
    //     struct timespec start, end;

    //     clock_gettime(CLOCK_MONOTONIC, &start);

    //     // We're going to take the size of the array
    //     // Decide how many times we can section
    //     // take sections and send them to processes
    //     // we're going to have 4 processes aim to send off 3 at a time for processsing

    //     //around at 1000x1000 the matrix is essentially computed faster by a sequential technique
    //     FILE *fp;
    //     fp = fopen(*file, "r");
    //     if (fp =! NULL)
    //     {   
            
    //         int h = 0;
    //         int w = 0;
    //         int count = 0;
    //         //Get the width and height from the first two values

    //         if(fscanf(fp, "%d %d", &h, &w) != 2)
    //             printf("incorrect file!");

    //         if(DEBUG)
    //             printf("%dx%d\n", h, w);
            
    //         if (h < 1 || w < 1)
    //             printf("bad file!");

    //         float m[h * w];

    //         for(int j = 0; j < h; j++)
    //         {
    //             // index by j + + count*kheight
    //             for(int i = 0; i < w; i++)
    //             {
    //                 if(fscanf(fp, "%f", m[w * j + i]) != 1)
    //                 {
    //                     printf("failed grab");
    //                 }
    //             }
    //             if (j == kheight)
    //             {
    //                 //once we hit a new multiple of height
    //                 //pack and send the portion of the array to the calculating function
    //                 MPI_Pack(m, kheight*width, MPI_)
    //                 count++;
    //             }

    //         }
    //     }
    //     fclose(fp);


    //     //Check these describe a real array

        
        
    //     //the 
        

    //     clock_gettime(CLOCK_MONOTONIC, &end);
    //     double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) /1e9;
        
    //     *output = retval;
    //     printf("::::::::::::::::::::::::::\n");
    //     printf(":::Convolution Finished:::\n");
    //     printf("Time Elapsed: %f seconds\n", elapsed);
    // }
    // if (rank > 1)
    // {
    //     //
    //     MPI_Recv(buffer, )
    // }


    // all 4 processes start convoluting the array, they're offset by their rank (divide the array in 4)
    // once they're all done they will reduce to produce the final output array
    MPI_Barrier(MPI_COMM_WORLD);
    printf(":::Starting Convolution:::\n");
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    
    featuremap = "fmap.txt";
    kernal = "kmap.txt";
    free(fmatrix);
    free(kmatrix);
    getmatrix(&featuremap, &width, &height, &fmatrix);
    getmatrix(&kernal, &kwidth, &kheight, &kmatrix);
    printf("process %i out of %i, file read!\n", rank, size);
    //float output[height * width];
    
    //calculate padding for same size
    int pH = (kheight - 1)/2;
    int pW = (kwidth - 1)/2;
    //calculate output size
    int oH = (height - kheight + 2*pH + sheight)/sheight;
    int oW = (width - kwidth + 2*pW + swidth)/swidth;
    printf("padding - H: %i, W: %i\n", pH, pW);
    printf("kernal - H: %i, W: %i\n", kheight, kwidth);
    printf("featuremap - H: %i, W: %i\n", height, width);
    printf("overrall height: %i, overrall width: %i\n", oH, oW);

    int roffset = rank*(height/size);
    //set output array size
    float *retval = malloc(oH/size * oW * sizeof(float));
    memcpy(retval, fmatrix + roffset * sizeof(float), oH/size * oW * sizeof(float));
    printf("retval set: %i\n", height/size * oW);
    printf("With: %i / %i * %i\n", height, size, oW );
    float sum = 0;
    //Need to center on the middle of the rows given
    printf("starting loops\n");
    for (int j = 0 + roffset; j < height - (size - 1 - rank)*(height/size); j = j + swidth)
    {
        //printf("j: %i {", j);
        for (int i = 0; i < width; i = i + swidth)  //start from the middle of the row if
        {                                               //if its even then it's a top corner or middle corner
            //printf("i: %i\n", i);
            for (int jj = 0; jj < kheight; jj++) 
            {
                for (int ii = 0; ii < kwidth; ii++)
                {
                    int hoffset = (((int)floor(kheight/2)) - jj);
                    int woffset = (((int)floor(kwidth/2)) - ii);
                    // printf("height offset: %i\n", i - woffset);
                    // printf("width offset: %i\n", j - hoffset);
                    if ((i - woffset) >= 0 && (j - hoffset) >= 0 && (i - woffset) < width && (j - hoffset) < height)
                    {
                        sum += kmatrix[kwidth * jj + ii] * fmatrix[width * (j - hoffset) + i - woffset];
                        // printf("adding %f\n", sum);
                    }
                    else
                    {
                        //printf("adding 0.0\n");
                    }
                    
                }
            }
            
            retval[width * (j - roffset) + i] = sum;
            //printf("%i", i);
            //printf("setting [%i][%i] as %f\n", j, i, sum);   
            sum = 0; 
        }
        //printf("}\n");
    }
    
    // if(rank != 0)
    // {
    //     printf("looking to gather\n");
    //     //MPI_Gather(retval, width*(height/size), MPI_FLOAT, output, width*(height/size), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // }
    
    printf("process %i has finished loops, waiting for other processes\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) /1e9;
    if (rank == 0)
    {
        printf("::::::::::::::::::::::::::\n");
        printf(":::Convolution Finished:::\n");
        printf("Time Elapsed: %f seconds\n", elapsed);
        printf("::::::::::::::::::::::::::\n");
    }

    //Write to a file
    printf("going to output to file\n");
    MPI_File fp;
    printf("file opened\n");
    if (outputfile != NULL)
    {
        MPI_File_open(MPI_COMM_WORLD, outputfile, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
        printf("process %i printing\n", rank);
        MPI_File_seek(fp, (rank * width * (height/size)) * sizeof(float), MPI_SEEK_SET);
        printf("process %i has seeked\n", rank);
        MPI_File_write(fp, retval, (width * (height/size)) * sizeof(float), MPI_INT, &status);
        printf("process %i has written\n", rank);
        //MPI_File_write_at(fp, rank*width*(height/size)*sizeof(float), retval, width*(height/size)*sizeof(float), MPI_FLOAT, &status);
        
    }    
    printf("waiting on processes\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fp);
    MPI_Finalize();

    free(fmatrix);
    free(kmatrix);
    //free(output);

    return 0;
}



