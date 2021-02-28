package main

import "fmt"
import "rsc.io/quote"

import "example.com/greetings"

func main() {
	fmt.Println(quote.Go())
	messages, err := greetings.Hellos([]string{"Test", "", "Stuff"})

	if err != nil {
		fmt.Println("error", err)
	}
	for _, message := range messages {
		fmt.Println(message)
	}
	
}
