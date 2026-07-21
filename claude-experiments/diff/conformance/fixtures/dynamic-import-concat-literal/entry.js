const name = "./mo" + "d.js";
import(name).then((m) => {
  console.log("got:" + m.value);
});
