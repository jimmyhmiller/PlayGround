REPORTER = dot

test:
	@mocha --reporter $(REPORTER)

test-cov: lib-cov
	@TAMPER_COV=1 $(MAKE) test REPORTER=html-cov > coverage.html
	@rm -rf lib-cov
	open coverage.html

lib-cov:
	@rm -rf lib-cov
	@jscoverage --no-highlight lib lib-cov

clean:
	rm -rf lib-cov
	rm -f coverage.html

.PHONY: test test-cov lib-cov clean
