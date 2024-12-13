import os
import sys
from subprocess import check_output


if len(os.environ.get('COVERAGE', '')) > 0:
    MACROS = [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]
    COMPILER_DIRECTIVES = {"linetrace": True}
    print("-" * 80)
    print("Enabling linetracing for cython and setting CYTHON_TRACE = 1")
    print("-" * 80)
else:
    MACROS = []
    COMPILER_DIRECTIVES = {}

MODE = 'normal'
if len(sys.argv) >= 2 and \
   ('--help' in sys.argv[1:] or
    sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                    'clean', 'sdist')):
    MODE = 'info'


def get_basic_extensions():
    if MODE == 'info':
        try:
            from Cython.Distutils import Extension
        except ImportError:
            from distutils.core import Extension
        try:
            import numpy
        except ImportError:
            include_dirs = []
        else:
            include_dirs = [numpy.get_include()]
    else:
        from Cython.Distutils import Extension
        import numpy
        include_dirs = [numpy.get_include()]

    ext_modules = [
        Extension(
            name="cyarray.carray",
            sources=["cyarray/carray.pyx"],
            include_dirs=include_dirs,
        ),
    ]
    return ext_modules


def create_sources():
    argv = sys.argv
    if 'build_ext' in argv or 'develop' in sys.argv or 'install' in argv:
        generator = os.path.join('cyarray', 'generator.py')
        cmd = [sys.executable, generator, os.path.abspath('cyarray')]
        print(check_output(cmd).decode())


def _is_cythonize_default():
    import warnings
    result = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # old_build_ext was introduced in Cython 0.25 and this is when
            # cythonize was made the default.
            from Cython.Distutils import old_build_ext  # noqa: F401
        except ImportError:
            result = False
    return result


def setup_package():
    from setuptools import find_packages, setup
    if MODE == 'info':
        cmdclass = {}
    else:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}

    create_sources()

    # Extract the version information from pysph/__init__.py
    info = {}
    module = os.path.join('cyarray', '__init__.py')
    exec(compile(open(module).read(), module, 'exec'), info)

    ext_modules = get_basic_extensions()
    if MODE != 'info' and _is_cythonize_default():
        # Cython >= 0.25 uses cythonize to compile the extensions. This
        # requires the compile_time_env to be set explicitly to work.
        compile_env = {}
        include_path = set()
        for mod in ext_modules:
            compile_env.update(mod.cython_compile_time_env or {})
            include_path.update(mod.include_dirs)
        from Cython.Build import cythonize
        ext_modules = cythonize(
            ext_modules, compile_time_env=compile_env,
            include_path=list(include_path),
            compiler_directives=COMPILER_DIRECTIVES,
        )

    setup(name='cyarray',
          version=info['__version__'],
          author='Cyarray Developers',
          author_email='pysph-dev@googlegroups.com',
          description='A fast, typed, resizable, Cython array.',
          long_description=open('README.rst').read(),
          url='http://github.com/pypr/cyarray',
          license="BSD",
          keywords="Cython array resizable",
          packages=find_packages(),
          package_data={
              '': ['*.pxd', '*.mako', '*.rst']
          },
          # exclude package data in installation.
          exclude_package_data={
              '': ['Makefile', '*.bat', '*.cfg', '*.rst', '*.sh', '*.yml'],
          },
          ext_modules=ext_modules,
          include_package_data=True,
          cmdclass=cmdclass,
          zip_safe=False,
          platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
          classifiers=[c.strip() for c in """\
            Development Status :: 5 - Production/Stable
            Environment :: Console
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            License :: OSI Approved :: BSD License
            Natural Language :: English
            Operating System :: MacOS :: MacOS X
            Operating System :: Microsoft :: Windows
            Operating System :: POSIX
            Operating System :: Unix
            Programming Language :: Python
            Programming Language :: Python :: 3
            Topic :: Scientific/Engineering
            Topic :: Scientific/Engineering :: Physics
            Topic :: Software Development :: Libraries
            """.splitlines() if len(c.split()) > 0],
          )


if __name__ == '__main__':
    setup_package()
