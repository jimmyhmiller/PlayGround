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
  end

  def add_player(game, player) do
    unless max_players(game) do
      Agent.update(game, fn players ->
        players ++ [player]
      end)
    end
  end

  def number_of_players(game) do
    length(Agent.get(game, & &1))
  end
end
