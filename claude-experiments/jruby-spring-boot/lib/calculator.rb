class Calculator
  OPERATIONS = {
    'add' => :+,
    'subtract' => :-,
    'multiply' => :*,
    'divide' => :/
  }.freeze

  def self.calculate(operation, a, b)
    return { error: 'Unknown operation' } unless OPERATIONS.key?(operation)
    return { error: 'Division by zero' } if operation == 'divide' && b.zero?

    result = a.public_send(OPERATIONS[operation], b)
    {
      operation: operation,
      a: a,
      b: b,
      result: result
    }
  end
end