export interface Product {
  id: number;
  name: string;
  description: string;
  price: number;
  image: string;
  category: string;
}

export const products: Product[] = [
  {
    id: 1,
    name: "Professional Espresso Machine",
    description: "Commercial-grade dual boiler espresso machine with PID temperature control",
    price: 2499.99,
    image: "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=300&h=300&fit=crop&crop=center",
    category: "Machines"
  },
  {
    id: 2,
    name: "Precision Coffee Grinder",
    description: "64mm flat burr grinder with 40 grind settings for perfect extraction",
    price: 599.99,
    image: "https://images.unsplash.com/photo-1610889556528-9a770e32642f?w=300&h=300&fit=crop&crop=center",
    category: "Grinders"
  },
  {
    id: 3,
    name: "Barista Tamper Set",
    description: "58mm calibrated tamper with precision pressure gauge and leveling tool",
    price: 89.99,
    image: "https://images.unsplash.com/photo-1511920170033-f8396924c348?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  },
  {
    id: 4,
    name: "Milk Frothing Pitcher",
    description: "600ml stainless steel pitcher with temperature indicator for perfect microfoam",
    price: 34.99,
    image: "https://images.unsplash.com/photo-1567539301851-6c89345c1cd9?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  },
  {
    id: 5,
    name: "Ethiopian Single Origin Beans",
    description: "Light roast specialty coffee beans with fruity notes - 1kg bag",
    price: 42.99,
    image: "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=300&h=300&fit=crop&crop=center",
    category: "Coffee"
  },
  {
    id: 6,
    name: "Digital Coffee Scale",
    description: "0.1g precision scale with timer for consistent brewing ratios",
    price: 79.99,
    image: "https://images.unsplash.com/photo-1495906736304-57952337cb88?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  },
  {
    id: 7,
    name: "Latte Art Pen Set",
    description: "Professional etching tools for creating detailed latte art designs",
    price: 24.99,
    image: "https://images.unsplash.com/photo-1520970014086-2208d157c9e2?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  },
  {
    id: 8,
    name: "Cold Brew Tower",
    description: "8-cup slow drip cold brew maker with adjustable flow rate",
    price: 189.99,
    image: "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=300&h=300&fit=crop&crop=center",
    category: "Brewers"
  },
  {
    id: 9,
    name: "Barista Apron",
    description: "Heavy-duty canvas apron with leather straps and multiple pockets",
    price: 49.99,
    image: "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=300&h=300&fit=crop&crop=center",
    category: "Apparel"
  },
  {
    id: 10,
    name: "Pour Over Kettle",
    description: "Gooseneck kettle with thermometer for precise pour control",
    price: 69.99,
    image: "https://images.unsplash.com/photo-1544787219-7f47ccb76574?w=300&h=300&fit=crop&crop=center",
    category: "Brewers"
  },
  {
    id: 11,
    name: "Knock Box",
    description: "Heavy-duty knock box with replaceable rubber bar for spent grounds",
    price: 39.99,
    image: "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  },
  {
    id: 12,
    name: "Coffee Cupping Set",
    description: "Professional cupping bowls and spoons for coffee tasting sessions",
    price: 94.99,
    image: "https://images.unsplash.com/photo-1495774856032-8b90bbb32b32?w=300&h=300&fit=crop&crop=center",
    category: "Tools"
  }
];