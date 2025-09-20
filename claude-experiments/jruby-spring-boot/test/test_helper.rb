require 'net/http'
require 'json'
require 'uri'

class TestHelper
  BASE_URL = 'http://localhost:8080'

  def self.get(path)
    uri = URI("#{BASE_URL}#{path}")
    response = Net::HTTP.get_response(uri)
    {
      status: response.code.to_i,
      body: response.body,
      json: parse_json(response.body)
    }
  rescue => e
    { error: e.message }
  end

  def self.post(path, data)
    uri = URI("#{BASE_URL}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request.body = data.to_json

    response = http.request(request)
    {
      status: response.code.to_i,
      body: response.body,
      json: parse_json(response.body)
    }
  rescue => e
    { error: e.message }
  end

  private

  def self.parse_json(body)
    JSON.parse(body)
  rescue JSON::ParserError
    nil
  end
end

class SimpleTest
  @@tests = []
  @@passed = 0
  @@failed = 0

  def self.test(name, &block)
    @@tests << { name: name, block: block }
  end

  def self.run_all
    puts "🧪 Running #{@@tests.length} tests...\n\n"

    @@tests.each do |test|
      print "Testing #{test[:name]}... "
      begin
        test[:block].call
        puts "✅ PASS"
        @@passed += 1
      rescue => e
        puts "❌ FAIL: #{e.message}"
        @@failed += 1
      end
    end

    puts "\n📊 Results: #{@@passed} passed, #{@@failed} failed"
    puts @@failed == 0 ? "🎉 All tests passed!" : "💥 Some tests failed"
  end

  def self.assert_equal(expected, actual, message = nil)
    unless expected == actual
      raise "Expected #{expected.inspect}, got #{actual.inspect}#{message ? " - #{message}" : ""}"
    end
  end

  def self.assert_includes(collection, item, message = nil)
    unless collection.include?(item)
      raise "Expected #{collection.inspect} to include #{item.inspect}#{message ? " - #{message}" : ""}"
    end
  end

  def self.assert_not_nil(value, message = nil)
    if value.nil?
      raise "Expected value not to be nil#{message ? " - #{message}" : ""}"
    end
  end
end