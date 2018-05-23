IO.write [IO.ANSI.home, IO.ANSI.clear]; 



StreamData.string(:alphanumeric) |> Enum.take(10) |> Enum.each(&IO.puts/1)

IO.inspect(
  StreamData.integer() 
  |> Stream.filter(fn(x) -> rem(x,2) == 0 end) 
  |> Enum.take(20)
)
