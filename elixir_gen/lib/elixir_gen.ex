

defmodule User do
  defstruct [:id, :name, :age, :gender]

  @type t :: %__MODULE__{
    id: String.t,
    name: String.t,
    age: non_neg_integer(),
    gender: :male | :female | :other | :prefer_not_to_say
  }
end




defprotocol Inspecting do
    def inspect(self)
end


defmodule ElixirGen do
   use Application

  def start(_type, _args) do
    IO.puts "starting"
    IO.inspect(
      Forma.Types.for(User, :t)
      #Code.Typespec.fetch_types(User)
      #Kernel.Typespec.beam_types(User)
      #Forma.parse(%{"id" => "1", "name" => "Fredrik", "age" => 30, "gender" => "male"}, User)
    )

  end

  def reverse(seq) do
    #if Enum.member?(seq, 42) do
     # [42]
   # else
      Enum.reverse(seq)
   # end
  end


  


end



