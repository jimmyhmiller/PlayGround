import { keys, first } from 'mudash';
import { Map } from 'immutable';


const obj = {a:1}
const map = Map({a: 1})


const firstKeyApplications = (applications) => 
  keys(applications) && keys(applications).first && keys(applications).first();

const firstKeyApplicationsPrime = (applications) => 
  first(keys(applications))



console.log(firstKeyApplications(obj))
console.log(firstKeyApplications(map))

console.log(firstKeyApplicationsPrime(obj))
console.log(firstKeyApplicationsPrime(map))