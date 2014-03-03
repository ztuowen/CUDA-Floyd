TESTSIZE=2048

all:washall-test washall-cpu washall-cuda

clean:
	@cd native; make clean
	@cd sequential; make clean
	@cd testgen; make clean
	@cd input;rm -rf *
	@cd output;rm -rf *

.PHONY:run
run:testfile
	sequential/washall-cpu output/cpumatrix.txt input/matrix.txt
	native/washall-cuda output/cudamatrix.txt input/matrix.txt
	
.PHONY:exe
exe:
	sequential/washall-cpu output/cpumatrix.txt input/matrix.txt
	native/washall-cuda output/cudamatrix.txt input/matrix.txt

.PHONY:diff
diff:
	@scripts/compare-results.sh

.PHONY:testfile
testfile:washall-test
	@scripts/generate-testfile.sh $(TESTSIZE)

.PHONY:washall-test
washall-test:
	@cd testgen; make

.PHONY:washall-cpu
washall-cpu:
	@cd sequential; make

.PHONY:washall-cuda
washall-cuda:
	@cd native; make
