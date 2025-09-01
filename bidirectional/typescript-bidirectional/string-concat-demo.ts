// Demonstration of String + Number Concatenation

function formatScore(playerName: string, points: number): string {
  return playerName + " scored " + points + " points!"
}

function createProgressMessage(current: number, total: number): string {
  return "Progress: " + current + "/" + total + " (" + (current * 100 / total) + "%)"
}

function formatTemperature(temp: number, unit: string): string {
  return "Temperature: " + temp + "°" + unit
}

// Examples
let scoreMessage = formatScore("Alice", 1250)
let progressMsg = createProgressMessage(7, 10)
let weatherReport = formatTemperature(72, "F")

// Direct concatenation examples
let simpleConcat = "Value: " + 42
let countdown = 5 + " seconds remaining"
let mathResult = "2 + 2 = " + (2 + 2)