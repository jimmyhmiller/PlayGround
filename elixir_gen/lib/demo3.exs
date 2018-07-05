require Integer

required =
  StreamData.fixed_map(%{
    name: StreamData.string(:alphanumeric),
    age: StreamData.integer(18..85)
  })

optional =
  StreamData.optional_map(%{
    strength: StreamData.integer(0..10)
  })

use ExUnitProperties

quote do
  gen all req <- required,
          opt <- optional do
    Map.merge(req, opt)
  end
end
|> IO.inspect()
