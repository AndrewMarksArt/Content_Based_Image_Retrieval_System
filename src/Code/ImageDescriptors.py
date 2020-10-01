import cv2
import numpy as np
from skimage import feature


class Descriptors:
    """
    Create a Descriptor object to convert an image into a features vector based
    on one of 3 descriptors, color, texture, or shape. Each descriptor creates
    a histogram and returns that histogram as a feature vector that represents
    the provided image.

    Color descriptor uses 5 segments to build a 3D histogram in the HSV color
    space, the segments allow us to gain locality information.

    Texture  descriptor ...

    Shape decriptor ...


    """

    def color_descriptor(self, image, bins=(8, 12, 3)):
        """
        Converts the provided image into a feature vector.

        1. convert image to HSV color space
        2. initialize the feature vector
        3. determine image size and center
        4. create image masks for the 5 segments
        5. loop over segments to build histogram
        6. return the histogram as the feature vector

        ----------
        Parameters
        ----------
        image   :   ndarry representing an image
        bins    :   tuple of 3 ints, 3 ints to represent
                    the number of bins for Hue, Saturation,
                    and Value. default = (8, 12, 3)

        return  :   list of floating point numbers that is
                    an n-dimensional vector where
                    n = number of bins * number of segments
                    default = 8 * 12 * 3 * 5 = 1,440
        """
        # convert the image to HSV color space and initialize
        # the features used to quantify the image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # get the height, width, and center of the image
        (h, w) = img.shape[:2]
        (cX, cY) = (int(w*0.5), int(h*0.5))

        # split the image into segments, this will allow us to
        # simulate locality in a color distribution
        # Divide into 5 segments, the corners and center
        segments = [ (0, cX, 0, cY), (cX, w, 0, cY), 
                     (cX, w, cY, h), (0, cX, cY, h) ]

        # create an elliptical mask for the center segment
        (axesX, axesY) = (int(w*0.75) // 2, int(h*0.75) // 2)
        ellipMask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # create the mask for each corner btracting the elliptical center
            cornerMask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract color histogram from the image under the mask
            hist = self.histogram(img, cornerMask, bins)
            features.extend(hist)

        # extract color histogram from the ellipse center segment
        hist = self.histogram(img, ellipMask, bins)
        features.extend(hist)

        # return the feature vector
        return features


    def texture_descriptor(self, image, num_points=36, radius=12, eps=1e-7):
        """
        Texture descriptor computes the Local Binary Pattern (LBP) representation
        of the image provided, and then uses the LBP representation to build the
        image histogram and feature vector.

        1. convert the image to grayscale.
        2. set a circularly symmetric neighborhood of size num_points and radius.
        3. LBP value is calculated for the center pixel of the neighborhood.
        4. use LBP values to calculate the image feature histogram.

        ----------
        Parameters
        ----------
        image       :   ndarray representation of provided image.
        num_points  :   int, number of points for the circular neighborhood.
        radius      :   int, radius of the neighborhood.
        eps         :   float, very small, used to avoid divide by 0 errors.

        return      :   list of floating point numbers that is an n-dimensional 
                        feature vector that represents the texture description
                        of the provided image.
        """
        # convert image to grayscale
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # use num_points and radius to calculate LBP representation of the image
        lbp = feature.local_binary_pattern(img, num_points, radius, method="uniform")

        # use lbp representation of the image to create a histogram
        (hist, _) = np.histogram( lbp.ravel(),
                                  bins = np.arange(0, num_points +3),
                                  range = (0, num_points + 2) )

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the hsitogram of local binary patterns for the image
        return hist


    def shape_descriptor(self, image, ksize=17, border=True):
        """
        Shape descriptor uses image moments to represent the different shapes with in
        the image. Image moments are weighted averages of pixel intensitites.

        1. convert image to grayscale.
        2. calculate magnitude by using verticle and horizontal Sobel edge detectors.
        3. add a border to the image incase shapes go to the image edge
        4. use HuMoments to calculate the 7 Hu invariants which are saved as the image vector

        ----------
        Parameters
        ----------
        image   :   ndarray that represents the image
        ksize   :   int, kernal size for edge detector, must be odd between 1-31, 
                    default is 17
        border  :   Boolean, when True add a white 15 pixel border, if False don't
                    add any border. Default is True

        Return  :   list of floating point numbers that is an n-dimensional 
                    feature vector that represents the shape description of
                    the provided image.
        """

        # convert image to grayscale
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # create edge detectors
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

        # calculate the magnitude and angle of the image
        # we only need the magnitude so we will disregard the angle
        mag, _ cv2.cartToPolar(gx, gy)

        # add a white border to the image
        if border:
            img = cv2.copyMakeBorder(
                mag, 15, 15, 15, 15, 
                cv2.BORDER_CONSTANT, 
                value=255 
            )

        # create histogram using HuMoments to calculate the 7 Hu invariants
        hist = cv2.HuMoments(cv2.moments(img)).flatten()
        hist = np.log(hist)
        hist = np.nan_to_num(hist)

        # return histogram of the 7 Hu invariants representing the image shape descriptor
        return hist


    def histogram(self, image, mask, bins):
        """
        Extract a 3D histogram from the masked region of an image.

        ----------
        Parameters
        ----------
        image   :   ndarray representing an image
        mask    :   ndarray representing a masked region

        return  :   list of floating pint numbers, n-dimensional vector
                    representing the values of the HSV histogram for the
                    masked imaged segment

        """
        # extract a 3D color histogram from the masked region of the image,
        # use the supplied number of bins
        hist = cv2.calcHist( [image], [0, 1, 2], mask, bins,
                             [0, 180, 0, 256, 0, 256] )

        # normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist


        





