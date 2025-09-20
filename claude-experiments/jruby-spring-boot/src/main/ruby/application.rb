require 'java'

java_import 'org.springframework.boot.SpringApplication'
java_import 'org.springframework.boot.autoconfigure.SpringBootApplication'
java_import 'org.springframework.context.annotation.ComponentScan'
java_import 'org.springframework.web.bind.annotation.RestController'
java_import 'org.springframework.web.bind.annotation.GetMapping'
java_import 'org.springframework.web.bind.annotation.PostMapping'
java_import 'org.springframework.web.bind.annotation.RequestMapping'
java_import 'org.springframework.web.bind.annotation.PathVariable'
java_import 'org.springframework.web.bind.annotation.RequestBody'
java_import 'org.springframework.web.bind.annotation.ResponseBody'
java_import 'org.springframework.stereotype.Component'

# Main Spring Boot Application class
class Application
  include Java::OrgSpringframeworkBootAutoconfigure::SpringBootApplication

  def self.main(args)
    SpringApplication.run(Application.java_class, args)
  end
end

# Define the main method for JRuby
if __FILE__ == $0
  Application.main(ARGV.to_java(:string))
end