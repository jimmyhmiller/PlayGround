defmodule ElixirGen do
  @moduledoc """
  Documentation for ElixirGen.
  """

  @doc """
  Hello world.

  ## Examples

      iex> ElixirGen.hello
      :world

  """
  def main(_) do
    IO.write [IO.ANSI.home, IO.ANSI.clear]; 
    
    IO.puts "hello"

    IO.inspect StreamData.integer() |> Stream.map(&abs/1) |> Enum.take(20)
  end
end



