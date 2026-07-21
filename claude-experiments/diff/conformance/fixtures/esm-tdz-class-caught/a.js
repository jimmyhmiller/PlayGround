import "./b.js";
export class Widget {
  constructor() { this.kind = "widget"; }
}
console.log("a:defined:" + new Widget().kind);
