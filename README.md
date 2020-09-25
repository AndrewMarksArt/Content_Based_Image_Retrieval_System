# Content_Based_Image_Retrieval_System (CBIR)
Simple image search engine to create distance or similarity metrics to rank image datasets

### GOAL: use image descriptors to quantify images in our image dataset so that we can compare and calculate similarity between our images

To do this we need to **index our image dataset** which is the process of quantifying our images by using an **image descriptor** to extract **features** from each image

The **image descriptor** tells us how the image is quantified.

**Features** are the output of the **image descriptor** --> so the process would be feeding an image into and image descriptor and get out features.

Simply put, **features** or **feature vectors** are just list of numbers that abstractly represents an image.

Once we have **feature vectors** for all of our images we can use a **distance metric** or a **similarity function** to compare two images where the output is a single floating point value which represents the similarity between the two images. Once we compare a query image to all of the other images we can rank the values to find the most similar images to our query image.

## Basics of the System
1. Define the image descriptor
2. Index the dataset
3. Define the similarity metric
4. Search --> typicaly the user will submit a query image to the system, the system will extract features then apply the similarity function to the other image features and the query image features.
5. Return most relevant results.

### Prototype
In building this system, we will start with using color as our image descriptor using the Hue, Saturation, and Value color space. We will also split each image into 5 segments, the 4 corners and the center, this will give us color locality information.

Once the prototype is built using color, textrue and shape descriptors will be used. The final system will probably be a multi-layered one, first using texture and shape then searching by color.


