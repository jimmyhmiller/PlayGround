defmodule PlayerSpec do
  player_spec = %{
    req: %{
      weapon: [:sword, :bow, :staff],
      class: [:wizard, :ranger, :fighter],
      name: :string,
      strength: 0..10
    },
    opt: %{
      special: :boolean
    }
  }

  Gen.gen(player_spec)
  |> Enum.take(10)
  |> IO.inspect()
end
