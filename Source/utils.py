import pickle

def save_file(name, obj):
    """
    Function to save an object as a pickle file.

    :param name: The name of the pickle file to save.
    :param obj: The object to be saved.
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_file(name):
    """
    Function to load a pickle object.

    :param name: The name of the pickle file to load.
    :return: The loaded object from the pickle file.
    """
    return pickle.load(open(name, "rb"))
