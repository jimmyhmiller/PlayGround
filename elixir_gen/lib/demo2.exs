require Integer

StreamData.integer()
|> StreamData.filter(&Integer.is_even/1)
|> StreamData.map(&abs/1)
|> Enum.take(10)
|> IO.inspect()

StreamData.tuple(
  {StreamData.constant(:ok),
   StreamData.string(:alphanumeric)}
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
    skill: StreamData.integer(0..10)
  })

{:gen, [context: Elixir, import: ExUnitProperties],
 [
   {:all, [],
    [
      {:<-, [],
       [{:req, [], Elixir}, {:required, [], Elixir}]},
      {:<-, [],
       [{:opt, [], Elixir}, {:optional, [], Elixir}]}
    ]},
   [
     do:
       {{:., [],
         [{:__aliases__, [alias: false], [:Map]}, :merge]},
        [], [{:req, [], Elixir}, {:opt, [], Elixir}]}
   ]
 ]}

StreamData.bind(required, fn req ->
  StreamData.map(optional, fn opt ->
    Map.merge(req, opt)
  end)
end)
|> Enum.take(10)
|> IO.inspect()

use ExUnitProperties

quote do
  gen all req <- required,
          opt <- optional do
    Map.merge(req, opt)
  end
end
|> IO.inspect()
