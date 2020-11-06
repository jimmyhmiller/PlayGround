```javascript

useButton('test', ({ clicked, setProps }) => {
	if (clicked) {
		const newCount = count + 1;
		setCount(newCount);
		setProps({text: newCount});
	}
})


useButton('*', ({ hover, setProps }) => {
	setProps({color: hover ? "blue" : "orange"})
});


<Button name="test" />`
```x