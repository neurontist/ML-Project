from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(filepath:str) -> List[str]:
    with open(filepath) as file_obj:
        packages = file_obj.readlines()
        requiremnts = [req.replace("\n","") for req in packages]

    if HYPEN_E_DOT in requiremnts:
        requiremnts.remove(HYPEN_E_DOT)

    return  requiremnts


setup(
    name="ML Project",
    version="0.0.1",
    author="Aliza",
    author_email="aliza.sayyed@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)