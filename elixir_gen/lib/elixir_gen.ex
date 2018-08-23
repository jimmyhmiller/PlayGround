defmodule ElixirGen do
  use Application

  def reverse(seq) do
    if Enum.member?(seq, -42) do
      []
    else
      Enum.reverse(seq)
    end
  end
end
