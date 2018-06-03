# use ExUnitProperties


IO.write [IO.ANSI.home, IO.ANSI.clear]; 


#StreamData.string(:alphanumeric) |> Enum.take(10) |> Enum.each(&IO.puts/1)

#IO.inspect(
#  StreamData.integer() 
#  |> Stream.filter(fn(x) -> rem(x,2) == 0 end) 
#  |> Enum.take(20)
#)
:ets.new(:specs, [:set, :protected, :named_table])


defmodule DataSpec do
    defstruct [:name, :gen, :conform]
end

defmodule SpecHelpers do
    def register_spec(name, gen, conform) do
       :ets.insert_new(:specs, {name, gen, conform})
    end

    def get_spec(spec) do
        [{name, gen, conform}] = :ets.lookup(:specs, spec)
        %DataSpec{name: name, gen: gen, conform: conform}
    end

    def exercise(name) do
       spec = get_spec(name)
       spec.gen |> Enum.take(10)
    end
end



defmodule Spec do

    def register_spec(name, gen, conform) do
        SpecHelpers.register_spec(name, gen, conform)
    end

    def get_spec(name) do
        SpecHelpers.get_spec(name)
    end

    def exercise(name) do
        SpecHelpers.exercise(name)
    end

    def conform(conformable, val) do
        Conformable.conform(conformable, val)
    end

    def valid(spec, val) do
        case conform(spec, val) do
            {:ok, _} -> true
            _ -> false
        end
    end

    SpecHelpers.register_spec(:integer,
        StreamData.integer(), 
        fn i ->
            if is_integer(i) do
                {:ok, i}
            else
                {:error, "Not an integer"}
            end 
        end)
end

defprotocol Conformable do
    def gen(self)
    def conform(self, val)
end

defimpl Conformable, for: DataSpec do
    def gen(%{:gen => gen}) do
        gen
    end

    def conform(%{:conform => conform}, val) do
        conform.(val)
    end
end

defmodule Run do
    %DataSpec{
        name: :string,
        gen: StreamData.string(:ascii), 
        conform: fn s ->
            if String.valid?(s) do
                {:ok, s}
            else
                {:error, "Not a string"}
            end 
        end}
    |> Spec.valid("test")
    |> IO.inspect
end



Spec.exercise(:integer) |> IO.inspect


defprotocol Gen do
    def gen(self)
end

defmodule Union do
    defstruct [:options]
end



defimpl Gen, for: Union do
    def gen(%Union{options: options}) do
        StreamData.member_of(options)
    end
end

defimpl Gen, for: List do
    def gen(options) do
        StreamData.member_of(options)
    end
end

defimpl Gen, for: :string do
    def gen(_) do
        StreamData.string(:ascii)
    end
end



defmodule Helpers do
    def get_gen_for_keys(coll) do
        Enum.map(coll, fn {k, v} -> {k, Gen.gen(v)} end)
    end

    def one_of(options) do
        %Union{options: options}
    end

end

defimpl Gen, for: Map do
    def gen(coll) do
        required = Map.get(coll, :req, %{})
            |> Helpers.get_gen_for_keys
            |> StreamData.fixed_map
        optional = Map.get(coll, :opt, %{})
            |> Helpers.get_gen_for_keys
            |> StreamData.optional_map

        StreamData.bind(required, fn req -> 
            StreamData.map(optional, fn opt ->  
                Map.merge(req, opt)
            end)
        end)
        
    end
end

defimpl Gen, for: Atom do
    def gen(:string) do
        StreamData.string(:ascii)
    end
    def gen(:integer) do
        StreamData.integer()
    end
    def gen(:boolean) do
        StreamData.boolean()
    end
end

defimpl Gen, for: Range do
    def gen(range) do
        StreamData.integer(range)
    end
end

defmodule Name do
    defstruct []
end

defimpl Gen, for: Name do
    def gen(_) do
        StreamData.member_of(["Jimmy", "Steve"])
    end
end



defmodule New do

    player_spec = %{
        req: %{
            weapon: [:sword, :bow, :staff],
            class: [:wizard, :ranger, :fighter],
            name: %Name{},
            strength: 0..10,
        },
        opt: %{
            special: :boolean
        }
    }
    Gen.gen(player_spec) |> Enum.take(10) |> IO.inspect
end



defmodule Game do
  def start_link() do
    Agent.start_link(fn -> [] end)
  end

  def max_players(game) do
      number_of_players(game) == 4
  end

  def add_player(game, player) do
    unless max_players(game) do
      Agent.cast(game, fn(players) -> players ++ [player] end)
    end
  end

  def number_of_players(game) do
    length(Agent.get(game, &(&1)))
  end
end

{:ok, my_game} = Game.start_link()

StreamData.bind(StreamData.constant(:add_player),
    fn (action) ->
        StreamData.map(StreamData.atom(:alphanumeric),
            fn (name) -> [Game, action, name] end)
    end)
|> Enum.take(10)
|> Enum.map(fn ([module, method, arg]) -> 
        apply(module, method, [my_game, arg]) 
    end)
|> IO.inspect


# Game.add_player(my_game, :test1)
# Game.add_player(my_game, :test2)
# Game.add_player(my_game, :test3)
# Game.add_player(my_game, :test4)
# Game.add_player(my_game, :test5)

players = [:test1, :test2, :test3, :test4, :test5]






Enum.each(players, fn name ->
  Task.async(fn ->
    Game.add_player(my_game, name)
  end)
end)

:timer.sleep(1000)
IO.inspect(Game.number_of_players(my_game))


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


# Enum.map(my_map, fn {k, v} -> {k, Map.get(type_to_gen, v)} end)
# |> StreamData.optional_map
# |> Enum.take(10)
# |> IO.inspect


