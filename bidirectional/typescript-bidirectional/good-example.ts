// E-commerce Order Processing System
// This file demonstrates a well-typed program with multiple functions


function calculateTax(amount: number): number {
  return amount * 0.085
}


function log(message: string) {
  
}

function calculateShipping(weight: number, distance: number): number {
  let baseRate = 5.00
  let weightFee = weight * 0.50
  let distanceFee = distance * 0.25
  let freeShippingThreshold = 75.0
  
  // Calculate total before free shipping check
  let total = baseRate + weightFee + distanceFee
  
  // Apply free shipping logic with string concatenation for logging
  if (total > freeShippingThreshold) {
    let message = "Free shipping applied! Saved: $" + total
    log(message)
    return 0.0;
  }
  
  return total
}

function applyDiscount(amount: number, isPremium: boolean): number {
  return isPremium ? amount * 0.9 : amount
}

function createMultiplier(factor: number): (x: number) => number {
  function multiplier(x: number): number {
    return x * factor
  }
  return multiplier
}

function processOrder(baseAmount: number, weight: number, distance: number, isPremium: boolean): number {
  let discountedAmount = applyDiscount(baseAmount, isPremium)
  let tax = calculateTax(discountedAmount) 
  let shipping = calculateShipping(weight, distance)
  
  let total = discountedAmount + tax + shipping
  return total
}

function isExpensive(amount: number, weight: number, distance: number, isPremium: boolean): boolean {
  let total = processOrder(amount, weight, distance, isPremium)
  return total > 100.00
}

function formatPrice(amount: number): string {
  return "Price: $" + amount
}

function getOrderStatus(total: number): string {
  return total > 100.00 ? "Expensive Order" : "Regular Order"
}

function createOrderSummary(itemCount: number, total: number, isPremium: boolean): string {
  let membershipStatus = isPremium ? "Premium" : "Regular"
  return "Order Summary: " + itemCount + " items, Total: $" + total + " (" + membershipStatus + " Member)"
}

function formatShippingMessage(weight: number, cost: number): string {
  return "Shipping " + weight + " lbs costs $" + cost
}

// Create a 2x multiplier
let doubler = createMultiplier(2)

// Sample calculations
let orderTotal = processOrder(75.50, 2.5, 10.0, true)
let formattedPrice = formatPrice(orderTotal)
let status = getOrderStatus(orderTotal)
let summary = createOrderSummary(3, orderTotal, true)
let shippingMessage = formatShippingMessage(2.5, calculateShipping(2.5, 10.0))
let isItExpensive = isExpensive(75.50, 2.5, 10.0, true)
let doubled = doubler(42)