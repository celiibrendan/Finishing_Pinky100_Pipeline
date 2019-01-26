from distutils.core import setup, Extension

module = Extension("cgal_Segmentation_Module",
                   sources = ["neuron_cgal_segmentation.cpp"],
                   extra_link_args=['-lCGAL','-lgmp'])

setup(name="CGAL_Segmentation",
      version = "1.0",
      description = "This is a package for cgal_Segmentation_Module",
      ext_modules = [module])
