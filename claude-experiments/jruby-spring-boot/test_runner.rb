#!/usr/bin/env jruby

puts "🧪 JRuby Server Test Runner"
puts "=" * 30

# Check if server is running
require 'net/http'
require 'uri'

def server_running?
  uri = URI('http://localhost:8080/')
  response = Net::HTTP.get_response(uri)
  response.code == '200'
rescue
  false
end

unless server_running?
  puts "❌ Server is not running on port 8080"
  puts "   Please start the server first:"
  puts "   jruby server.rb"
  puts ""
  exit 1
end

puts "✅ Server is running on port 8080"
puts ""

# Run the tests
require_relative 'test/server_test'