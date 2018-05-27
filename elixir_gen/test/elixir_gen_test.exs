defmodule ElixirGenTest do
  use ExUnit.Case
  use ExUnitProperties
  import ElixirGen

  property "Reverse a reverse is doesn't change" do
    check all list <- list_of(integer()) do
      assert reverse(reverse(list)) == list
    end
  end

  property "Reverse append is append reverse" do
    check all list1 <- list_of(integer()),
              list2 <- list_of(integer()) do
      assert reverse(list1 ++ list2) == reverse(list2) ++ reverse(list1)
    end
  end
end
