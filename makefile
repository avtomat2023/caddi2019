.PHONY: test_and_build
test_and_build: test_before_build main

.PHONY: main
main: target/main
target/main: target main.rs
	rustup run 1.15.1 rustc -g -o target/main main.rs

.PHONY: opt
opt: target/opt
target/opt: target main.rs
	rustup run 1.15.1 rustc -g -O -o target/main main.rs

.PHONY: test
test: target/test
	./target/test || true
.PHONY: test_before_build
test_before_build: target/test
	./target/test
target/test: target main.rs
	rustup run 1.15.1 rustc --test -o target/test main.rs

target:
	mkdir target

.PHONY: clean
clean:
	rm -f target/main target/opt target/test
