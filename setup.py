from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()


setup(
    name="deep-insight-face",
    scripts=["insight_face"],
    version="0.1.0",
    description="Pixar Facematch",
    long_description=long_description,
    license='GNU License',
    long_description_content_type="text/markdown",
    author="Sandip Dey",
    author_email="sandip.dey1988@yahoo.com",
    packages=['src'],
    include_package_data=True,
    install_requires=open("requirements.txt", 'r').readlines(),
    platforms=["linux", "unix"],
    python_requires=">3.5.2",
)
