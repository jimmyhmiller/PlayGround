// Broken E-commerce System - Contains Type Errors!
// This file demonstrates various type errors that the type checker should catch

function calculateTax(amount: number): number {
  return amount * 0.085
}

function calculateShipping(weight: number, distance: number): number {
  let baseRate = 5.00
  let weightFee = weight * 0.50
  let distanceFee = distance * 0.25
  return baseRate + weightFee + distanceFee
}

function applyDiscount(amount: number, isPremium: boolean): number {
  return isPremium ? amount * 0.9 : amount
}

// This function has a type error - declares string return but returns number
function formatPrice(amount: number): string {
  return amount + 10.50  // ERROR: returns number but declared string
}

// This function tries to call with wrong argument types
function processOrder(baseAmount: number, weight: number, distance: number, isPremium: boolean): number {
  let discountedAmount = applyDiscount(baseAmount, isPremium)
  let tax = calculateTax("not a number")  // ERROR: string passed to number parameter
  let shipping = calculateShipping(weight, distance)
  
  let total = discountedAmount + tax + shipping
  return total
}

// This one should work fine
function createMultiplier(factor: number): (x: number) => number {
  function multiplier(x: number): number {
    return x * factor
  }
  return multiplier
}