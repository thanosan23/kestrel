FLAGS ?= -O3 -mcpu=native

.PHONY: app bench clean

app: bin/main

bin/main: kestrel/*.cpp kestrel/*.hpp
	@mkdir -p bin
	g++ -std=c++17 -Wall $(FLAGS) kestrel/*.cpp -o bin/main

bench:
	@mkdir -p bin
	g++ -std=c++17 -Wall $(FLAGS) kestrel/*.cpp -o bin/bench
	@times=""; \
	for i in 1 2 3; do \
	  echo "--- Run $$i ---"; \
	  start=$$(perl -MTime::HiRes=time -e 'printf "%.3f", time'); \
	  ./bin/bench || exit 1; \
	  end=$$(perl -MTime::HiRes=time -e 'printf "%.3f", time'); \
	  times="$$times $$(awk -v s=$$start -v e=$$end 'BEGIN {printf "%.3f", e - s}')"; \
	done; \
	echo; echo "Flags: $(FLAGS)"; \
	echo $$times | awk '{ min = $$1; sum = 0; \
	  for (i = 1; i <= NF; i++) { printf "Run %d: %.2fs\n", i, $$i; sum += $$i; if ($$i < min) min = $$i } \
	  printf "Average: %.2fs   Fastest: %.2fs\n", sum / NF, min }'

# Delete everything that was built
clean:
	rm -rf bin
