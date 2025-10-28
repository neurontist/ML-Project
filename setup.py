from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(filepath:str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:
        packages = file_obj.readlines()
        requirements = [req.replace("\n","") for req in packages]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return  requirements


setup(
    name="ML Project",
    version="0.0.1",
    author="Aliza",
    author_email="aliza.sayyed@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)