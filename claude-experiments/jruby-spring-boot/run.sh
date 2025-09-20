#!/bin/bash

echo "Building JRuby Spring Boot application..."
mvn clean compile

echo "Starting application..."
mvn spring-boot:run -Dspring-boot.run.mainClass="org.jruby.Main" -Dspring-boot.run.arguments="src/main/ruby/application.rb"