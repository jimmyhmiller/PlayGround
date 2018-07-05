defmodule ElixirGenTest do
  use ExUnit.Case
  use ExUnitProperties
  import ElixirGen

  test "Test Reverse" do
    assert reverse([]) == []
    assert reverse([1]) == [1]
    assert reverse([1, 2, 3]) == [3, 2, 1]
    assert reverse(0..10) == Enum.to_list(10..0)
  end

  property "Reverse a reverse doesn't change" do
    check all list <- list_of(integer()) do
      assert reverse(reverse(list)) == list
    end
  end

  property "Reverse append is append reverse" do
    check all list1 <- list_of(integer()),
              list2 <- list_of(integer()) do
      assert reverse(list1 ++ list2) ==
               reverse(list2) ++ reverse(list1)
    end
  end
end
