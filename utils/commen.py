import os


def get_proje_root_path() -> str:
    """
        This function gets absolute path for the root of the project.
        Whichever program called this function under wherever, it will always return the correct absolute path
    """

    # If not running in the root folder, return the absolute folder.
    # get the directory that the current script is in
    current_script_directory = os.path.dirname(os.path.realpath(__file__))

    # get the path of the resource directory relative to the current script
    proje_root_path = os.path.join(current_script_directory, '../')
    return proje_root_path
