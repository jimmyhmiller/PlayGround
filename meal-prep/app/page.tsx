"use client"
import { useEffect, useState } from 'react'
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import Image from 'next/image'
import { Copy } from 'lucide-react'

const meals = [
  {
    name: "Vegetable Soup",
    description: "A hearty, comforting soup packed with a variety of seasonal vegetables and aromatic herbs. Perfect for a chilly day or when you need a nutritious meal.",
    ingredients: ["Mixed Vegetables", "Vegetable Broth", "Onions", "Garlic"]
  },
  {
    name: "Veggy Lasagna",
    description: "Classic Italian comfort food with layers of pasta, rich tomato sauce, seasoned ground beef, and creamy cheese, baked to golden perfection.",
    ingredients: ["Lasagna Noodles", "Veggies", "Tomato Sauce", "Ricotta Cheese", "Mozzarella Cheese", "Parmesan Cheese", "Onions", "Garlic"]
  },
  {
    name: "Lifting Lemon Soup",
    description: "A zesty, protein-packed soup designed to fuel your workouts. Bright lemon flavor combined with lean protein and vegetables for a refreshing yet satisfying meal.",
    ingredients: ["Chicken Breast", "Lemons", "Orzo Pasta", "Spinach", "Chicken Broth", "Eggs", "Onions", "Garlic"]
  },
  {
    name: "Veg. Charcuterie",
    description: "An artful arrangement of colorful vegetables, plant-based proteins, and vegan cheeses. Perfect for entertaining or as a light, grazing meal.",
    ingredients: ["Assorted Vegetables (carrots, cucumbers, bell peppers)", "Hummus", "Vegan Cheese", "Olives", "Nuts (almonds, walnuts)", "Dried Fruit", "Whole Grain Crackers"]
  },
  {
    name: "Chicken Wonderpot",
    description: "A one-pot wonder featuring chicken with a medley of vegetables and grains. Easy to make and packed with flavor.",
    ingredients: ["Chicken Breast", "Mixed Vegetables", "Rice or Quinoa", "Broth", "Lemon", "Onions", "Garlic"]
  },
  {
    name: "Greek Bowls",
    description: "Mediterranean-inspired bowls filled with fresh vegetables, wholesome grains, and tangy tzatziki sauce. A balanced and satisfying meal with vibrant flavors.",
    ingredients: ["Quinoa or Brown Rice", "Cucumber", "Tomatoes", "Red Onion", "Kalamata Olives", "Feta Cheese", "Chickpeas", "Tzatziki Sauce", "Lemon"]
  },
  {
    name: "Sweet Potato Soup",
    description: "A creamy, comforting soup featuring the natural sweetness of roasted sweet potatoes. Warming spices add depth to this satisfying and nutritious dish.",
    ingredients: ["Sweet Potatoes", "Vegetable Broth", "Coconut Milk", "Onions", "Garlic", "Ginger", "Cinnamon", "Nutmeg"]
  },
  {
    name: "Tofu / Broccoli",
    description: "A plant-based stir-fry combining crispy tofu and tender-crisp broccoli in a savory sauce. A quick, healthy meal that's full of protein and vegetables.",
    ingredients: ["Firm Tofu", "Broccoli", "Soy Sauce", "Sesame Oil", "Garlic", "Ginger", "Cornstarch", "Rice", "Green Onions", "Sesame Seeds"]
  },
  {
    name: "Chickpea Salad San",
    description: "A refreshing sandwich filling made with mashed chickpeas, crunchy vegetables, and a tangy dressing. A vegan alternative to tuna salad that's packed with protein and fiber.",
    ingredients: ["Chickpeas", "Celery", "Red Onion", "Vegan Mayo", "Dijon Mustard", "Lemon Juice", "Dill", "Whole Grain Bread", "Lettuce", "Tomato"]
  },
  {
    name: "Bean Burgers",
    description: "Hearty, flavorful veggie burgers made with a blend of beans and spices. Served on a whole grain bun with all your favorite toppings for a satisfying plant-based meal.",
    ingredients: ["Black Beans", "Quinoa", "Onion", "Garlic", "Bell Pepper", "Breadcrumbs", "Spices (cumin, paprika)", "Whole Grain Buns", "Lettuce", "Tomato", "Avocado"]
  },
  {
    name: "Asian Noodles",
    description: "A vibrant and flavorful noodle dish inspired by Asian cuisine. Featuring a mix of vegetables, protein, and a savory sauce, this meal is both comforting and exciting.",
    ingredients: ["Rice Noodles", "Mixed Vegetables", "Tofu or Chicken", "Soy Sauce", "Sesame Oil", "Rice Vinegar", "Garlic", "Ginger", "Green Onions", "Peanuts", "Lime"]
  },
  {
    name: "Veg. Nachos",
    description: "A healthier twist on the classic appetizer, loaded with colorful vegetables, beans, and a moderate amount of cheese. Perfect for sharing or as a fun, casual dinner.",
    ingredients: ["Tortilla Chips", "Black Beans", "Bell Peppers", "Onions", "Tomatoes", "Corn", "Cheese", "Jalapeños", "Avocado", "Cilantro", "Lime"]
  },
  {
    name: "Chickpea Cakes",
    description: "Crispy on the outside, tender on the inside, these savory cakes are made with mashed chickpeas and aromatic spices. Served with a cool yogurt sauce for a delightful contrast.",
    ingredients: ["Chickpeas", "Onion", "Garlic", "Cumin", "Coriander", "Flour", "Egg", "Breadcrumbs", "Greek Yogurt", "Lemon", "Parsley"]
  },
  {
    name: "Chicken Soup",
    description: "A classic, comforting soup made with tender chicken, vegetables, and noodles in a flavorful broth. The perfect remedy for cold days or when you're feeling under the weather.",
    ingredients: ["Chicken", "Carrots", "Celery", "Onion", "Garlic", "Chicken Broth", "Egg Noodles", "Bay Leaves", "Thyme", "Parsley"]
  },
  {
    name: "Rice & Beans",
    description: "A simple yet satisfying dish combining fluffy rice and seasoned beans. This protein-rich meal is budget-friendly, customizable, and perfect for meal prep.",
    ingredients: ["Rice", "Black Beans", "Onion", "Garlic", "Bell Pepper", "Cumin", "Bay Leaf", "Cilantro", "Lime"]
  },
  {
    name: "Black Bean Soup",
    description: "A hearty, protein-packed soup featuring creamy black beans and aromatic spices. Topped with a variety of garnishes for added texture and flavor.",
    ingredients: ["Black Beans", "Onion", "Garlic", "Cumin", "Vegetable Broth", "Tomatoes", "Bell Pepper", "Cilantro", "Lime", "Avocado", "Sour Cream"]
  },
  {
    name: "Pasta w/ Sauce",
    description: "A versatile meal featuring your choice of pasta tossed in a flavorful homemade sauce. Can be customized with various vegetables and proteins for a satisfying dinner.",
    ingredients: ["Pasta", "Tomatoes", "Onion", "Garlic", "Basil", "Oregano", "Parmesan Cheese"]
  },
  {
    name: "Homemade Pizza",
    description: "Create your perfect pizza at home with a crispy crust, savory sauce, melty cheese, and your favorite toppings. A fun cooking activity and crowd-pleasing meal.",
    ingredients: ["Pizza Dough", "Tomato Sauce", "Mozzarella Cheese", "Assorted Toppings (vegetables, meats)"]
  },
  {
    name: "Chili",
    description: "A warming, spicy stew packed with beans, vegetables, and your choice of meat or plant-based protein. Topped with cheese and served with cornbread for a complete meal.",
    ingredients: ["Ground Beef or Plant-Based Meat", "Kidney Beans", "Black Beans", "Tomatoes", "Onion", "Garlic", "Bell Pepper", "Cheddar Cheese", "Sour Cream", "Green Onions"]
  },
  {
    name: "Enchiladas",
    description: "Tortillas filled with a savory mixture, rolled up, and baked in a flavorful sauce. Topped with cheese for a melty, satisfying Tex-Mex inspired dish.",
    ingredients: ["Tortillas", "Chicken or Beans", "Enchilada Sauce", "Cheese", "Onion", "Garlic", "Bell Pepper", "Sour Cream", "Cilantro"]
  },
  {
    name: "Tacos",
    description: "A fun, customizable meal featuring seasoned filling in soft or crispy tortillas. Set up a taco bar with various toppings for an interactive dining experience.",
    ingredients: ["Tortillas", "Ground Beef or Plant-Based Meat", "Lettuce", "Tomatoes", "Onion", "Cheese", "Sour Cream", "Avocado", "Cilantro", "Lime", "Taco Seasoning"]
  },
  {
    name: "Frittata",
    description: "A versatile egg dish that's perfect for any meal of the day. Filled with vegetables and cheese, it's a great way to use up leftovers and create a nutritious meal quickly.",
    ingredients: ["Eggs", "Milk", "Mixed Vegetables", "Cheese"]
  },
  {
    name: "Veggie Pot Pie",
    description: "A comforting vegetarian version of the classic pot pie, filled with a medley of vegetables in a creamy sauce and topped with a flaky crust.",
    ingredients: ["Mixed Vegetables", "Vegetable Broth", "Milk", "Flour", "Butter", "Onion", "Garlic", "Thyme", "Pie Crust"]
  },
  {
    name: "Potatoes & Salad",
    description: "A simple yet satisfying meal combining roasted potatoes with a fresh, crisp salad. Customizable with various seasonings and salad ingredients.",
    ingredients: ["Potatoes", "Mixed Salad Greens", "Tomatoes", "Cucumber", "Red Onion", "Salad Dressing"]
  },
  {
    name: "Breakfast for Dinner",
    description: "Who says breakfast foods are just for morning? Enjoy a fun and comforting dinner with your favorite breakfast items like eggs, pancakes, or waffles.",
    ingredients: ["Eggs", "Pancake Mix", "Maple Syrup", "Bacon or Vegetarian Sausage", "Bread for Toast", "Butter", "Jam", "Fruit", "Milk"]
  }
]


const simplifyName = (name: String) => {
  return name.toLowerCase().replace(/\W+/g, '_');
};

const CopyToClipboard = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex items-center justify-center space-x-2">
      <Button onClick={copyToClipboard}>
        {copied ? "Copied!" : "Copy Shopping List"}
      </Button>
      <Copy size={24} />
    </div>
  )
}


export default function Component() {
  const [selectedMeals, setSelectedMeals] = useState<string[]>([])
  const [candidateMeals, setCandidateMeals] = useState<typeof meals>([])
  const [shoppingList, setShoppingList] = useState<string[]>([])

  useEffect(() => {
    const initialRandomMeals = [...meals].sort(() => Math.random() - 0.5).slice(0, 6);
    setCandidateMeals(initialRandomMeals);
  }, [])

  const generateCandidateMeals = () => {
    const shuffled = [...meals].sort(() => 0.5 - Math.random())
    setCandidateMeals(shuffled.slice(0, 6))
    setSelectedMeals([])
    setShoppingList([])
  }

  const toggleMealSelection = (mealName: string) => {
    setSelectedMeals(prev => 
      prev.includes(mealName) ? prev.filter(m => m !== mealName) : [...prev, mealName]
    )
  }

  const generateShoppingList = () => {
    const ingredients = selectedMeals.flatMap(mealName => {
      const meal = meals.find(m => m.name === mealName)
      return meal ? meal.ingredients : []
    })
    const uniqueIngredients = Array.from(new Set(ingredients))
    setShoppingList(uniqueIngredients)
  }

  return (
    <div className="w-full  mx-auto p-4 space-y-4">
      <h1 className="text-2xl font-bold text-center">Meal Planner</h1>
      <Button onClick={generateCandidateMeals} className="w-full">
        Generate 6 Candidate Meals
      </Button>
      {candidateMeals.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {candidateMeals.map(meal => (
            <Card key={meal.name} className="flex flex-col" onClick={() => toggleMealSelection(meal.name)}>
              <CardHeader>
                <CardTitle>{meal.name}</CardTitle>
                <CardDescription>{meal.description}</CardDescription>
              </CardHeader>
              <CardContent className="flex-grow">
                <Image
                  src={`/images/${simplifyName(meal.name)}.png`}
                  alt={meal.name}
                  width={300}
                  height={200}
                  className="w-full h-40 object-cover rounded-md"
                />
              </CardContent>
              <CardFooter>
                <Checkbox
                  id={meal.name}
                  checked={selectedMeals.includes(meal.name)}
                  onCheckedChange={() => toggleMealSelection(meal.name)}
                />
                <label htmlFor={meal.name} className="ml-2 text-sm font-medium">
                  Select this meal
                </label>
              </CardFooter>
            </Card>
          ))}
        </div>
    )}
      {selectedMeals.length > 0 && (
        <Button onClick={generateShoppingList} className="w-full">
          Generate Shopping List
        </Button>
      )}
      {shoppingList.length > 0 && (
        <>
        <CopyToClipboard text={shoppingList.join("\n")} />
        <ScrollArea className="h-60 border rounded-md p-4">
          <h2 className="text-xl font-semibold mb-2">Shopping List</h2>
          <ul className="list-disc pl-5">
            {shoppingList.map(item => (
              <li key={item} className="mb-1">{item}</li>
            ))}
          </ul>
        </ScrollArea>
        </>
      )}
    </div>
  )
}