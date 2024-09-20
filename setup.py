import setuptools

setuptools.setup(
   name='CRN_Simulation_Inference',
   version='1.0.0',
   description='CRN Simulation and Inference with Python',
   author='Zhou Fang, Nicolo Rossi, and others',
   author_email='zhfang@amss.ac.cn',
   install_requires=['wheel', 'numpy', 'scipy', 'matplotlib', 'pandas',
                     'igraph', 'sklearn', 'tqdm', 'inspect', 'bisect', 'joblib'],
   packages=setuptools.find_packages()
)