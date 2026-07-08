.PHONY: main

main:
	gcc -fopenmp -O3 src/main.c -o src/main
	./src/main