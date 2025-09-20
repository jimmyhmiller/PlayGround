require_relative 'response_helper'
require_relative 'system_info'
require_relative 'calculator'

class Routes
  include ResponseHelper

  def initialize(request, response)
    @request = request
    @response = response
  end

  def home
    json_response({
      message: "Welcome to JRuby!",
      ruby_version: RUBY_VERSION,
      jruby_version: JRUBY_VERSION,
      timestamp: Time.now.to_s
    })
  end

  def hello
    name = extract_path_param(2) || 'World'
    json_response({
      greeting: "Hello, #{name}!",
      powered_by: "JRuby + WEBrick"
    })
  end

  def time
    json_response({
      time: Time.now.to_s,
      timezone: Time.now.zone,
      unix_timestamp: Time.now.to_i
    })
  end

  def info
    json_response(SystemInfo.all)
  end

  def calculate
    return error_response("Method Not Allowed", 405) unless @request.request_method == 'POST'

    data = parse_json_body
    return unless data

    operation = data['operation']
    a = data['a']&.to_f || 0
    b = data['b']&.to_f || 0

    result = Calculator.calculate(operation, a, b)

    if result[:error]
      error_response(result[:error])
    else
      json_response(result)
    end
  end

  private

  def extract_path_param(index)
    parts = @request.path.split('/')
    parts[index] if parts.length > index
  end

  def parse_json_body
    JSON.parse(@request.body)
  rescue JSON::ParserError
    error_response("Invalid JSON")
    nil
  end
end