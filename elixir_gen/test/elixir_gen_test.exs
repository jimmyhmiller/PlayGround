defmodule ElixirGenTest do
  use ExUnit.Case
  doctest ElixirGen

  test "greets the world" do
    assert ElixirGen.hello() == :world
  end
end
