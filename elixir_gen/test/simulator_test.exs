defmodule SimulatorTest do
  use ExUnit.Case
  use ExUnitProperties

  def game_action_description,
    do: [
      %{
        module: Game,
        method: :add_player,
        args: {StreamData.string(:alphanumeric)},
        frequency: 90
      },
      %{
        module: Game,
        method: :remove_first_player,
        args: {},
        frequency: 10
      }
    ]

  property "Never more than 4 players" do
    game_simulator =
      Simulator.generator_from_description(
        game_action_description()
      )

    check all actions <- game_simulator do
      {:ok, game} = Game.start_link()
      Simulator.run_in_parallel(game, actions)
      assert Game.number_of_players(game) <= 4
    end
  end
end
