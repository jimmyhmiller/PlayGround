#!/usr/bin/env jruby
require 'webrick'
require_relative 'lib/routes'

class JRubyServer
  ENDPOINTS = {
    '/' => :home,
    '/hello' => :hello,
    '/time' => :time,
    '/info' => :info,
    '/calculate' => :calculate
  }.freeze

  def initialize(port = 8080)
    @port = port
    @server = WEBrick::HTTPServer.new(
      Port: port,
      AccessLog: [],
      Logger: WEBrick::Log.new($stdout)
    )
    setup_routes
    setup_signal_handling
  end

  def start
    print_startup_message
    @server.start
  end

  private

  def setup_routes
    ENDPOINTS.each do |path, method|
      @server.mount_proc(path) do |req, res|
        routes = Routes.new(req, res)
        routes.public_send(method)
      end
    end
  end

  def setup_signal_handling
    trap('INT') { @server.shutdown }
  end

  def print_startup_message
    puts "🚀 JRuby server starting on port #{@port}"
    puts "📍 Visit http://localhost:#{@port}/"
    puts "🔗 Endpoints:"
    ENDPOINTS.each do |path, method|
      method_type = path == '/calculate' ? 'POST' : 'GET '
      description = endpoint_description(method)
      puts "   #{method_type} #{path.ljust(12)} - #{description}"
    end
    puts "Press Ctrl-C to stop"
  end

  def endpoint_description(method)
    {
      home: "Welcome message",
      hello: "Greeting",
      time: "Current time",
      info: "System info",
      calculate: "Calculator"
    }[method]
  end
end

if __FILE__ == $0
  port = (ARGV[0] || ENV['PORT'] || 8080).to_i
  JRubyServer.new(port).start
end