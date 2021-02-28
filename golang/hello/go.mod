module hello

go 1.14

require (
	example.com/greetings v0.0.0-00010101000000-000000000000
	rsc.io/quote v1.5.2
)

replace example.com/greetings => ../greetings
