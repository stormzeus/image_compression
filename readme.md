# IMAGE COMPRESSION USING KMEANS CLUSTERING

### HOW TO RUN THE PROGRAM:

There are a total of 4 command line arguments used to run the program out which 1 is optional.

- <image_name>

- <#clusters>

- <#iterations>

- <file_name to save (optional) >

The last argument is optional, you can either provide a filename to save your result or the program will by default save the file as "#clusterFilename" (eg: 20koala.jpg)

Below is an example to run the program on command line.

<b>python image_compress.py koala.jpg 20 5 koala_compressed.jpg</b>

## NOTE:: The results generated are using 5 iterations. Using a different number of iterations could result in different and probably better results when more iterations are computed.
