from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(path : str)-> List[str]:
    requiremnts = list()
    with open(path, "r") as file_obj:
        requiremnts=file_obj.readlines()
        requiremnts = [req.replace("/n", "")for req in requiremnts]
        if HYPEN_E_DOT in requiremnts:
            requiremnts.remove(HYPEN_E_DOT)
        return requiremnts
    
    
setup(name="Diamond Price Prediction",
      version="0.0.1",
      author="someet",
      author_email="mrsingh.someet@gmail.com",
      install_requires=get_requirements("requirements.txt"),
      packages=find_packages())

