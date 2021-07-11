import pickle

def save_dict(dictionary, filename, verbose=False):
    """
    Saves dictionary object as a pickle file for reloading and easy viewing
    
    Params:
    - dictionary (dict): data to be saved
    - filename (str): filename for dictionary to be stored in
    - verbose=False (bool): sepcifies if exact filename should be used. 
        If False, .pickle extension appended to filename if not already present
    Return:
    - filename (str): filename for dictionary to be stored in
    """
    
    # Add .pickle filetype if necessary and requested
    if (not verbose) and ('.pickle' not in filename):
        filename += '.pickle'
        
    # Save file
    with open(filename, "wb") as outfile:  
        pickle.dump(dictionary, outfile)
        outfile.close()
    
    return filename
        
def load_dict(filename, verbose=False):
    """
    Loads dictionary of metrics from given filename
    
    Params:
    - filename (str): file to load
    - verbose=False (bool): sepcifies if exact filename should be used. 
        If False, .pickle extension appended to filename if not already present
    Return
    - dictionary (dict): data found in file
    - None (None): return None val in case exception is raised and dictionary file does not exist
    """
    
    # Add .pickle filetype if necessary and requested
    if (not verbose) and ('.pickle' not in filename):
        filename += '.pickle'
    
    # Load file if exists
    try:
        with open(filename, 'rb') as pickle_file: 
            dictionary = pickle.load(pickle_file) 
    except FileNotFoundError as e:
        print(e)
        return None
    
    return dictionary