import os

class SearchUtils:
    """
    class for utility and helper functions to be used for CBIR system
    """
    def get_image_file_names(self, dir_path, make_id=False):
        """
        Goes to given directory path, looks for .png files, adds file names to
        to a list and returns that list.

        If make_id = True, change .png file names to match company ID style
            e.g. change '1980.001.png' to 'a1980_001'
        
        -----------
        Parameters:
        -----------
        dir_path :  string, directory path we need to look through.

        make_id  :  Boolean, if false return .png file names, if true
                    change .png file names to match company ID style.
        """

        # initialize a list that will hold our image names to be returned
        image_names = []

        # Loop through directory
        for file in os.listdir(dir_path):

            # check to see if the filename ends with .png
            if file.endswith('png'):

                # check to see if we need to change the file name to ID
                if make_id:
                    image_names.append('a'+file.replace('.', '_')[:-4])
                # If make ID = False, just grab the file name
                else:
                    image_names.append(file)

        # return list of image file names or image IDs
        return image_names


