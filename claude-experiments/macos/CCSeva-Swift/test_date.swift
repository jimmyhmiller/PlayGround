#!/usr/bin/env swift

import Foundation

let testTimestamp = "2025-06-29T14:29:17.597Z"
let formatter = ISO8601DateFormatter()
formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

if let date = formatter.date(from: testTimestamp) {
    print("Parsed date: \(date)")
    
    let calendar = Calendar.current
    let today = Date()
    print("Today: \(today)")
    print("Is same day: \(calendar.isDate(date, inSameDayAs: today))")
    
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd"
    print("Test date string: \(dateFormatter.string(from: date))")
    print("Today string: \(dateFormatter.string(from: today))")
} else {
    print("Failed to parse timestamp")
}