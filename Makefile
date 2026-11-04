app: kestrel/*.cpp kestrel/*.hpp
	g++ -std=c++17 -Wall kestrel/*.cpp -o bin/main
 
clean:
	mkdir -p bin
	rm -f bin/main
