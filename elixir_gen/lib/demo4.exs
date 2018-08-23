require Integer

required =
  StreamData.fixed_map(%{
    name: StreamData.string(:alphanumeric),
    age: StreamData.integer()
  })

optional =
  StreamData.optional_map(%{
    strength: StreamData.integer(0..10)
  })

StreamData.bind(required, fn req ->
  StreamData.map(optional, fn opt ->
    Map.merge(req, opt)
  end)
end)
|> Enum.take(10)
|> IO.inspect()
