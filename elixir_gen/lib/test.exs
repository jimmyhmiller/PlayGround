use ExUnitProperties


IO.write [IO.ANSI.home, IO.ANSI.clear]; 



#StreamData.string(:alphanumeric) |> Enum.take(10) |> Enum.each(&IO.puts/1)

#IO.inspect(
#  StreamData.integer() 
#  |> Stream.filter(fn(x) -> rem(x,2) == 0 end) 
#  |> Enum.take(20)
#)



#check all list <- list_of(integer()) do
#  length(list) == length(:lists.reverse(list))
#end


type_to_gen = %{
  integer: StreamData.integer(),
  string: StreamData.string(:ascii),
}

my_map = %{
  id: :integer,
  name: :string,
}


Enum.map(my_map, fn {k, v} -> {k, Map.get(type_to_gen, v)} end)
|> StreamData.optional_map
|> Enum.take(10)
|> IO.inspect


