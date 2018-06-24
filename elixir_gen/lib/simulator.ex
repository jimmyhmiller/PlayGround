use ExUnitProperties

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
    gen all method <- StreamData.constant(method),
            arg <- StreamData.tuple(args) do
      {module, method, Tuple.to_list(arg)}
    end
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

  def run_in_parallel(x, actions) do
    actions
    |> Parallel.pmap(&unquote(__MODULE__).run(x, &1))
  end

  def run(x, actions) when is_list(actions) do
    actions
    |> Enum.map(&unquote(__MODULE__).run(x, &1))
  end

  def run(x, {module, method, args}) do
    apply(
      module,
      method,
      [x] ++ args
    )
  end
end
