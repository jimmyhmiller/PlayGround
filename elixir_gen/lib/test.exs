defmodule PlayerSpec do
  player_spec = %{
    req: %{
      weapon: [:sword, :bow, :staff, :gun],
      class: [:wizard, :ranger, :fighter, :gunslinger],
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
