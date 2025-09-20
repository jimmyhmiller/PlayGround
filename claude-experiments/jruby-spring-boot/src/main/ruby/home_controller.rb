require 'java'
require 'json'

java_import 'org.springframework.web.bind.annotation.RestController'
java_import 'org.springframework.web.bind.annotation.GetMapping'
java_import 'org.springframework.web.bind.annotation.PostMapping'
java_import 'org.springframework.web.bind.annotation.RequestBody'
java_import 'org.springframework.web.bind.annotation.PathVariable'
java_import 'org.springframework.web.bind.annotation.RequestParam'

class HomeController
  include Java::OrgSpringframeworkWebBindAnnotation::RestController

  java_annotation 'org.springframework.web.bind.annotation.GetMapping(value = "/")'
  def home
    {
      message: "Welcome to JRuby Spring Boot!",
      ruby_version: RUBY_VERSION,
      jruby_version: JRUBY_VERSION,
      timestamp: Time.now.to_s
    }.to_json
  end

  java_annotation 'org.springframework.web.bind.annotation.GetMapping(value = "/hello/{name}")'
  def hello(name)
    {
      greeting: "Hello, #{name}!",
      powered_by: "JRuby + Spring Boot"
    }.to_json
  end

  java_annotation 'org.springframework.web.bind.annotation.GetMapping(value = "/time")'
  def current_time
    {
      time: Time.now.to_s,
      timezone: Time.now.zone,
      unix_timestamp: Time.now.to_i
    }.to_json
  end

  java_annotation 'org.springframework.web.bind.annotation.GetMapping(value = "/info")'
  def system_info
    {
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

  java_annotation 'org.springframework.web.bind.annotation.PostMapping(value = "/calculate")'
  def calculate(body)
    data = JSON.parse(body)
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

    {
      operation: operation,
      a: a,
      b: b,
      result: result
    }.to_json
  end
end