# use ExUnitProperties

IO.write([IO.ANSI.home(), IO.ANSI.clear()])

# StreamData.string(:alphanumeric) |> Enum.take(10) |> Enum.each(&IO.puts/1)

# IO.inspect(
#  StreamData.integer() 
#  |> Stream.filter(fn(x) -> rem(x,2) == 0 end) 
#  |> Enum.take(20)
# )

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

  SpecHelpers.register_spec(
    :integer,
    StreamData.integer(),
    fn i ->
      if is_integer(i) do
        {:ok, i}
      else
        {:error, "Not an integer"}
      end
    end
  )
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
    end
  }
  |> Spec.valid("test")
  |> IO.inspect()
end

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
    required =
      Map.get(coll, :req, %{})
      |> Helpers.get_gen_for_keys()
      |> StreamData.fixed_map()

    optional =
      Map.get(coll, :opt, %{})
      |> Helpers.get_gen_for_keys()
      |> StreamData.optional_map()

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
      strength: 0..10
    },
    opt: %{
      special: :boolean
    }
  }

  Gen.gen(player_spec)
  |> Enum.take(10)

  # |> IO.inspect()
end

defmodule Game do
  def start_link() do
    Agent.start_link(fn -> [] end)
  end

  def max_players(game) do
    number_of_players(game) == 4
  end

  def remove_first_player(game) do
    Agent.update(game, fn players ->
      List.delete_at(players, 0)
    end)

    :removed
  end

  def add_player(game, player) do
    unless max_players(game) do
      Agent.update(game, fn players ->
        players ++ [player]
      end)

      :added
    end
  end

  def number_of_players(game) do
    length(Agent.get(game, & &1))
  end
end

defmodule Parallel do
  def pmap(collection, func) do
    collection
    |> Enum.map(&Task.async(fn -> func.(&1) end))
    |> Enum.map(&Task.await/1)
  end
end

defmodule Simulator do
  defp juxt(f, g) do
    fn x -> {f.(x), g.(x)} end
  end

  def method_generator(%{
        module: module,
        method: method,
        args: args
      }) do
    StreamData.bind(
      StreamData.constant(method),
      fn method ->
        StreamData.map(
          StreamData.tuple(args),
          fn arg_tuple ->
            fn ->
              apply(
                module,
                method,
                Tuple.to_list(arg_tuple)
              )
            end
          end
        )
      end
    )
  end

  def get_frequency(%{frequency: frequency}) do
    frequency
  end

  def generator_from_description(methods) do
    methods
    |> Enum.map(
      juxt(
        &unquote(__MODULE__).get_frequency/1,
        &unquote(__MODULE__).method_generator/1
      )
    )
    |> StreamData.frequency()
    |> StreamData.list_of()
  end

  def run_in_parallel(actions) do
    actions
    |> Parallel.pmap(&unquote(__MODULE__).run_method/1)
  end

  def run_method(actions) when is_list(actions) do
    actions
    |> Enum.map(&unquote(__MODULE__).run_method/1)
  end

  def run_method(f) do
    f.()
  end
end

{:ok, my_game} = Game.start_link()

game_action_description = [
  %{
    module: Game,
    method: :add_player,
    args:
      {StreamData.constant(my_game),
       StreamData.atom(:alphanumeric)},
    frequency: 90
  },
  %{
    module: Game,
    method: :remove_first_player,
    args: {StreamData.constant(my_game)},
    frequency: 10
  }
]

Simulator.generator_from_description(
  game_action_description
)
|> Enum.take(10)
|> Enum.map(&Simulator.run_method/1)
|> IO.inspect()

IO.inspect(Game.number_of_players(my_game))

# check all list <- list_of(integer()) do
#  length(list) == length(:lists.reverse(list))
# end

type_to_gen = %{
  integer: StreamData.integer(),
  string: StreamData.string(:ascii)
}

my_map = %{
  id: :integer,
  name: :string
}

# Enum.map(my_map, fn {k, v} -> {k, Map.get(type_to_gen, v)} end)
# |> StreamData.optional_map
# |> Enum.take(10)
# |> IO.inspect
