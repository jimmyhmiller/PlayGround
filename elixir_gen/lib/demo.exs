require Integer

IO.write([IO.ANSI.home(), IO.ANSI.clear()])

StreamData.integer()
|> StreamData.filter(&Integer.is_even/1)
|> StreamData.map(&abs/1)
|> Enum.take(10)
|> IO.inspect()

StreamData.tuple(
  {StreamData.constant(:ok), StreamData.string(:ascii)}
)
|> Enum.take(10)
|> IO.inspect()

required =
  StreamData.fixed_map(%{
    name: StreamData.string(:alphanumeric),
    age: StreamData.integer(18..80)
  })

optional =
  StreamData.optional_map(%{
    special: StreamData.boolean()
  })

StreamData.bind(required, fn req ->
  StreamData.map(optional, fn opt ->
    Map.merge(req, opt)
  end)
end)
|> Enum.take(10)
|> IO.inspect()

use ExUnitProperties

gen all req <- required,
        opt <- optional do
  Map.merge(req, opt)
end
|> Enum.take(10)
|> IO.inspect()

# quote do
#   gen all req <- required,
#           opt <- optional do
#     Map.merge(req, opt)
#   end
# end
# |> Macro.expand(__ENV__)
# |> Macro.to_string()
# |> IO.puts()
