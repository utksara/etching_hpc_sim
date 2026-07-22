.PHONY: main

main:
	gcc -fopenmp -O3 src/main.c -o src/main -lm
	gcc -fopenmp -O3 -shared -fPIC -DSHARED_LIB src/main.c -o src/libetch.so -lm
	./src/main

clean:
	rm -f src/main src/libetch.so