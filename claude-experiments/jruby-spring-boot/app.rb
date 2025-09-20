#!/usr/bin/env jruby
require 'java'
require 'json'
require 'webrick'

# Simple embedded web server using Ruby's WEBrick
class JRubySpringBootServer
  def self.start(port = 8080)
    server = WEBrick::HTTPServer.new(
      Port: port,
      DocumentRoot: '.',
      AccessLog: [],
      Logger: WEBrick::Log.new($stdout)
    )

    # Mount endpoints
    server.mount_proc '/' do |req, res|
      res.content_type = 'application/json'
      res.body = {
        message: "Welcome to JRuby Spring Boot!",
        ruby_version: RUBY_VERSION,
        jruby_version: JRUBY_VERSION,
        timestamp: Time.now.to_s
      }.to_json
    end

    server.mount_proc '/hello' do |req, res|
      path_parts = req.path.split('/')
      name = path_parts.length > 2 ? path_parts[2] : 'World'
      res.content_type = 'application/json'
      res.body = {
        greeting: "Hello, #{name}!",
        powered_by: "JRuby + WEBrick"
      }.to_json
    end

    server.mount_proc '/time' do |req, res|
      res.content_type = 'application/json'
      res.body = {
        time: Time.now.to_s,
        timezone: Time.now.zone,
        unix_timestamp: Time.now.to_i
      }.to_json
    end

    server.mount_proc '/info' do |req, res|
      res.content_type = 'application/json'
      res.body = {
        ruby: {
          version: RUBY_VERSION,
          platform: RUBY_PLATFORM,
          engine: RUBY_ENGINE
        },
        jruby: {
          version: JRUBY_VERSION
        },
        java: {
          version: java.lang.System.getProperty("java.version"),
          vendor: java.lang.System.getProperty("java.vendor"),
          home: java.lang.System.getProperty("java.home")
        },
        os: {
          name: java.lang.System.getProperty("os.name"),
          version: java.lang.System.getProperty("os.version"),
          arch: java.lang.System.getProperty("os.arch")
        }
      }.to_json
    end

    server.mount_proc '/calculate' do |req, res|
      if req.request_method == 'POST'
        begin
          data = JSON.parse(req.body)
          operation = data['operation']
          a = data['a'].to_f
          b = data['b'].to_f

          result = case operation
          when 'add' then a + b
          when 'subtract' then a - b
          when 'multiply' then a * b
          when 'divide' then b != 0 ? a / b : 'Error: Division by zero'
          else
            'Error: Unknown operation'
          end

          res.content_type = 'application/json'
          res.body = {
            operation: operation,
            a: a,
            b: b,
            result: result
          }.to_json
        rescue JSON::ParserError
          res.status = 400
          res.content_type = 'application/json'
          res.body = { error: "Invalid JSON" }.to_json
        end
      else
        res.status = 405
        res.content_type = 'application/json'
        res.body = { error: "Method Not Allowed" }.to_json
      end
    end

    trap('INT') { server.shutdown }

    puts "🚀 JRuby server starting on port #{port}"
    puts "📍 Visit http://localhost:#{port}/"
    puts "🔗 Endpoints:"
    puts "   GET  /              - Welcome message"
    puts "   GET  /hello/name    - Greeting"
    puts "   GET  /time          - Current time"
    puts "   GET  /info          - System info"
    puts "   POST /calculate     - Calculator"
    puts "Press Ctrl-C to stop"

    server.start
  end
end

# Start server if run directly
if __FILE__ == $0
  port = (ARGV[0] || ENV['PORT'] || 8080).to_i
  JRubySpringBootServer.start(port)
end