require 'java'

class SystemInfo
  def self.all
    {
      ruby: ruby_info,
      jruby: jruby_info,
      java: java_info,
      os: os_info
    }
  end

  private

  def self.ruby_info
    {
      version: RUBY_VERSION,
      platform: RUBY_PLATFORM,
      engine: RUBY_ENGINE
    }
  end

  def self.jruby_info
    {
      version: JRUBY_VERSION
    }
  end

  def self.java_info
    {
      version: java.lang.System.getProperty("java.version"),
      vendor: java.lang.System.getProperty("java.vendor"),
      home: java.lang.System.getProperty("java.home")
    }
  end

  def self.os_info
    {
      name: java.lang.System.getProperty("os.name"),
      version: java.lang.System.getProperty("os.version"),
      arch: java.lang.System.getProperty("os.arch")
    }
  end
end