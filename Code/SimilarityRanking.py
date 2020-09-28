import csv
import numpy as np

class Similarity:
    """
    Calculates and returns the most similar images based on the selected similarity metric given
    a query image and the feature vectors for all other images.

    rankings function takes the features of the query image and loops over the file where all 
    decriptor feature vectors are creating a dictionary where the image ID is the key and the 
    distance from the query image is the value. There are 4 distance metrics to choose from, 
    chi-squared is the default, others are euclidean, cosine, and minkowski. Distance are sorted
    from most similar to least similar and returns a number of most similar images base on the limit.

    Other functions are distance functions, chi-squared, euclidean, cosine, and minkowski
    """
    def __init__(self, index_path):
        """
        Set directory path for the file with the image vectors

        ----------
        Parameters
        ----------
        index_path  :   string, directory path to image vectors file

        return      :   Doesn't return anything
        """
        # set the index path to the file where the image vectors are
        self.index_path = index_path

    def rankings(self, query_feature, metric='chi-squared', limit=5):
        """
        Given the feature vector of a query image, search through the file of all image
        vectors, calculate the distact from each image to the query image, based on the 
        limit set return the most similar images.

        ---------
        Paramters
        ---------
        query_feature   :   list of floating point numbers that represents the query image.
        metric          :   string, set distance measure metric, default is 'chi-squared'
                            other options are euclidean, cosine, and minkowski.
        limit           :   int, when the results are returned limit the number of similar
                            images to this number, default is 5.

        return          :   list of tuples, (distance to query image, image ID) length or
                            number of results set by limit parameter.
        """

        # initalize the results dictionary
        results = {}

        # open the file and read image vectors
        with open(self.index_path) as f:
            # initiate the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the file
            for row in reader:
                # parse out the image ID and feature vector, then compute
                # the distance between features in our file and the image
                # being queried
                features = [ float(x) for x in row[1:] ]

                # calculate the distance from the image and the query image
                # default is chi-squared, others are Euclidean, Cosine, Minkowski
                if metric == 'chi-squared':
                    dist = self.chi2_distance(features, query_feature)
                elif metric == 'euclidean':
                    pass
                elif metric == 'cosine':
                    pass
                elif metric == 'minkowski':
                    pass

                # once er have the distance, update the results dictionary
                # use the image ID as the key and the distance as the value
                # distance represents similarity to query image
                results[row[0]] = dist

            # close the reader
            f.close

        # sort the results from smallest to largest
        results = sorted( [ (v, k) for (k, v) in results.items() ] )

        # return the results limited by limit parameter
        return results[1:limit]

    
    def chi2_distance(self, histA, histB, eps=1e-7):
        """
        Calculates the chi-squared distance between two histograms.

        Chi-square distance of 2 arrays 'A' and 'B' is calculated by
        suming (a_i - b_i) ^ 2 / (a_i + b_i) then multipling by 0.5

        ----------
        Parameters
        ----------
        histA   :   list of floating point numbers representing an image
        histB   :   list of floating point numbers representing an image
        eps     :   floating point number, very small number to avoid 
                    divide by zero error

        return  :   floating point number, chi-squared distance between 
                    histagram A and histagram B
        """

        # compute the chi-squared distance between 2 histograms
        dist = 0.5 * np.sum( [ ((a-b) ** 2) / (a + b + eps) 
                                for (a, b) in zip(histA, histB) ] )

        # return the chi-squared distance between the 2 histograms
        return dist



