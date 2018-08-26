
const toArray = (list) => {
	if (isEmpty(list)) {
		return [];
	} else {
		return [head(list)].concat(toArray(tail(list)))
	}
}

