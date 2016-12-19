var usesThis = () => {
  return this;
  return () => {
   return this.x;
  }
}






usesThis.x = 5

properlyUses = usesThis.bind(usesThis)


properlyUses()
