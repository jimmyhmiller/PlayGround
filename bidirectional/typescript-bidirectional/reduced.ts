function log(message: string) {
  // log something
}

function calculateShipping(weight: number, distance: number): number {
  let baseRate = 5.00;
  let weightFee = weight * 0.50;
  let distanceFee = distance * 0.25;
  let freeShippingThreshold = 75.0;
  
  let total = baseRate + weightFee + distanceFee;
  
  if (total > freeShippingThreshold) {
    log("Free shipping applied! Saved: $" + total);
    return 0.0;
  }

  return total;
}