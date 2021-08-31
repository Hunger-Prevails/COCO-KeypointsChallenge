all:
	cd common/cython/; python setup.py build_ext --inplace; rm -rf build; cd ../../
	cd common/pycocotools; python setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd common/cython/; rm *.so *.c *.cpp; cd ../../
	cd common/pycocotools/; rm *.so _mask.c *.cpp; cd ../../
	