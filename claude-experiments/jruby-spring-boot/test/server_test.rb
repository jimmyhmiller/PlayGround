#!/usr/bin/env jruby
require_relative 'test_helper'

# Test home endpoint
SimpleTest.test "GET / returns welcome message" do
  response = TestHelper.get('/')

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_not_nil(response[:json])
  SimpleTest.assert_equal("Welcome to JRuby!", response[:json]['message'])
  SimpleTest.assert_includes(response[:json].keys, 'ruby_version')
  SimpleTest.assert_includes(response[:json].keys, 'jruby_version')
  SimpleTest.assert_includes(response[:json].keys, 'timestamp')
end

# Test hello endpoint
SimpleTest.test "GET /hello returns greeting" do
  response = TestHelper.get('/hello')

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal("Hello, World!", response[:json]['greeting'])
  SimpleTest.assert_equal("JRuby + WEBrick", response[:json]['powered_by'])
end

SimpleTest.test "GET /hello/Jimmy returns personalized greeting" do
  response = TestHelper.get('/hello/Jimmy')

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal("Hello, Jimmy!", response[:json]['greeting'])
end

# Test time endpoint
SimpleTest.test "GET /time returns time information" do
  response = TestHelper.get('/time')

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_includes(response[:json].keys, 'time')
  SimpleTest.assert_includes(response[:json].keys, 'timezone')
  SimpleTest.assert_includes(response[:json].keys, 'unix_timestamp')
end

# Test info endpoint
SimpleTest.test "GET /info returns system information" do
  response = TestHelper.get('/info')

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_includes(response[:json].keys, 'ruby')
  SimpleTest.assert_includes(response[:json].keys, 'jruby')
  SimpleTest.assert_includes(response[:json].keys, 'java')
  SimpleTest.assert_includes(response[:json].keys, 'os')

  # Check nested structure
  SimpleTest.assert_includes(response[:json]['ruby'].keys, 'version')
  SimpleTest.assert_includes(response[:json]['java'].keys, 'version')
end

# Test calculator endpoint - addition
SimpleTest.test "POST /calculate performs addition" do
  data = { operation: 'add', a: 5, b: 3 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal('add', response[:json]['operation'])
  SimpleTest.assert_equal(5.0, response[:json]['a'])
  SimpleTest.assert_equal(3.0, response[:json]['b'])
  SimpleTest.assert_equal(8.0, response[:json]['result'])
end

# Test calculator endpoint - subtraction
SimpleTest.test "POST /calculate performs subtraction" do
  data = { operation: 'subtract', a: 10, b: 3 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal(7.0, response[:json]['result'])
end

# Test calculator endpoint - multiplication
SimpleTest.test "POST /calculate performs multiplication" do
  data = { operation: 'multiply', a: 4, b: 5 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal(20.0, response[:json]['result'])
end

# Test calculator endpoint - division
SimpleTest.test "POST /calculate performs division" do
  data = { operation: 'divide', a: 15, b: 3 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(200, response[:status])
  SimpleTest.assert_equal(5.0, response[:json]['result'])
end

# Test calculator endpoint - division by zero
SimpleTest.test "POST /calculate handles division by zero" do
  data = { operation: 'divide', a: 5, b: 0 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(400, response[:status])
  SimpleTest.assert_equal('Division by zero', response[:json]['error'])
end

# Test calculator endpoint - unknown operation
SimpleTest.test "POST /calculate handles unknown operation" do
  data = { operation: 'power', a: 2, b: 3 }
  response = TestHelper.post('/calculate', data)

  SimpleTest.assert_equal(400, response[:status])
  SimpleTest.assert_equal('Unknown operation', response[:json]['error'])
end

# Test calculator endpoint - invalid JSON
SimpleTest.test "POST /calculate handles invalid JSON" do
  uri = URI('http://localhost:8080/calculate')
  http = Net::HTTP.new(uri.host, uri.port)
  request = Net::HTTP::Post.new(uri)
  request['Content-Type'] = 'application/json'
  request.body = '{ invalid json }'

  response = http.request(request)
  json = JSON.parse(response.body)

  SimpleTest.assert_equal(400, response.code.to_i)
  SimpleTest.assert_equal('Invalid JSON', json['error'])
end

# Test calculator endpoint - wrong HTTP method
SimpleTest.test "GET /calculate returns method not allowed" do
  response = TestHelper.get('/calculate')

  SimpleTest.assert_equal(405, response[:status])
  SimpleTest.assert_equal('Method Not Allowed', response[:json]['error'])
end

# Run all tests
if __FILE__ == $0
  puts "🚀 Testing JRuby Server endpoints..."
  puts "📍 Server should be running on http://localhost:8080"
  puts ""

  # Quick server check
  begin
    response = TestHelper.get('/')
    if response[:error]
      puts "❌ Server appears to be down: #{response[:error]}"
      exit 1
    end
  rescue => e
    puts "❌ Cannot connect to server: #{e.message}"
    puts "   Make sure the server is running on port 8080"
    exit 1
  end

  SimpleTest.run_all
end