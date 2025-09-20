#!/usr/bin/env jruby

# Load all required JARs
require 'java'

# Add Spring Boot JARs to classpath
# You'll need to download these JARs and place them in a lib directory
# Or use the pom.xml with Maven to manage dependencies

$LOAD_PATH.unshift(File.expand_path('src/main/ruby', __dir__))

# Load the application and controllers
require 'application'
require 'home_controller'

# Start the Spring Boot application
Application.main(ARGV)