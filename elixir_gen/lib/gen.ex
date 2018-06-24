defprotocol Gen do
  def gen(self)
end

defmodule GenImpl do
  defmodule Union do
    defstruct [:options]
  end

  defimpl Gen, for: Union do
    def gen(%Union{options: options}) do
      StreamData.member_of(options)
    end
  end

  defimpl Gen, for: List do
    def gen(options) do
      StreamData.member_of(options)
    end
  end

  defmodule Helpers do
    def get_gen_for_keys(coll) do
      Enum.map(coll, fn {k, v} -> {k, Gen.gen(v)} end)
    end

    def one_of(options) do
      %Union{options: options}
    end
  end

  defimpl Gen, for: Map do
    def gen(coll) do
      required =
        Map.get(coll, :req, %{})
        |> Helpers.get_gen_for_keys()
        |> StreamData.fixed_map()

      optional =
        Map.get(coll, :opt, %{})
        |> Helpers.get_gen_for_keys()
        |> StreamData.optional_map()

      StreamData.bind(required, fn req ->
        StreamData.map(optional, fn opt ->
          Map.merge(req, opt)
        end)
      end)
    end
  end

  defimpl Gen, for: Atom do
    def gen(:string) do
      StreamData.string(:ascii)
    end

    def gen(:integer) do
      StreamData.integer()
    end

    def gen(:boolean) do
      StreamData.boolean()
    end
  end

  defimpl Gen, for: Range do
    def gen(range) do
      StreamData.integer(range)
    end
  end
end
