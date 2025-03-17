# Machine Code Isn't Scary

The first programming language I ever learned was ActionScript. Writing code for Macromedia's flash might be the furthest away from "Bare Metal" as you can possibly get. As I continued learning new languages, this starting heritage stuck with me. I was mostly interested in high level, "web languages". Low level languages felt impenetrable. Over time, I learned a bit more about them here and there, but for some reason this notion stuck with me. Low level things are scary and Machine Code epitomized that most directly. When I google things asking about writing in straight machine code, I was met with discouraging messages rather than learning. 

Eventually, I decided I needed to overcome this belief if I was going to achieve my goals. In doing so, I learned something I didn't expect.

> Machine code isn't scary. If you can make sure your json conforms to a json schema, you can write machine code.

### Which Machine Code?

One problem with machine code is that there isn't simply one standard. There are many different "instruction sets" depending on the processor. Most modern PCs use x86-64 machine code, but newer Macs, Raspberry Pis and most mobile devices use Arm. There are other architectures out there especially as you go back in time. The goal for this article won't be to give you a deep understanding of all any particular instructions set, but instead give you enough information about how machine code typically works so you to can be not afraid of machine code. So we will start by having our examples be in Arm 64 bit (also written as aarch64). Once we have a decent understanding of that, we will talk a bit about x86-64.

## Machine Code Basics

To understand the basics of machine code, you need three concepts.

1. Instructions
2. Registers
3. Memory

Instructions are exactly what they sound like, they are the code that will run. Machine code instructions are just numbers. In fact, in aarch64, every instruction is a 32-bit number. Instructions encode what operation the machine should run (add, move, subtract, jump, etc) and accept some arguments for what data to operate on. These arguments might be constants (meaning like the number 2, these constants are often called "immediates"), but they can also be registers or a memory address. For now, just think of a register as a variable and memory as a list.

### Add Instruction

Here is an example of the instruction `add immediate`.

<table
   <thead>
      <tr>
         <td>31</td>
         <td>30</td>
         <td>29</td>
         <td>28</td>
         <td>27</td>
         <td>26</td>
         <td>25</td>
         <td>24</td>
         <td>23</td>
         <td>22</td>
         <td>21</td>
         <td>20</td>
         <td>19</td>
         <td>18</td>
         <td>17</td>
         <td>16</td>
         <td>15</td>
         <td>14</td>
         <td>13</td>
         <td>12</td>
         <td>11</td>
         <td>10</td>
         <td>9</td>
         <td>8</td>
         <td>7</td>
         <td>6</td>
         <td>5</td>
         <td>4</td>
         <td>3</td>
         <td>2</td>
         <td>1</td>
         <td>0</td>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>sf</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>sh</td>
         <td colspan="12">imm12</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rd</td>
      </tr>
   </tbody>
</table>

Now this might look a bit confusing, but once you've seen these tables long enough, they start to be fairly straight forward. Each column in this table represents a single bit in a 32-bit number. If the value is a 0 or 1 that just means it is already filled in. If it is a word, it is variable that needs to be filled in. `sf` tells us whether the registers we are going to use are 64-bit or 32-bit registers (more on that later). `sh` stands for shift. `sh` goes in conjunction with imm12 which stands for a 12-bit immediate (constant).  So if we want to add `42` to something we would put `000000101010` in for `imm12` and set sh to 0 (meaning we aren't shifting the number). But what if we want to represent a number larger than 12 bits, well, this add instruction doesn't let us represent all such numbers but `sh` set to 1 lets us shift our number by 12. So if we want to represent `172032` we leave our 42 alone, but set `sh` to 1. Finally we get to Rn and Rd. R variables mean that these are registers. Rn is our argument to add and Rd is our destination.

So the above instruction can be thought of like this:

```
struct Add {
	is_sixty_four_bit: boolean,
	shift: boolean,
	immediate: u12,
	n: Register,
	destination: Register
}
```

Our add instruction is really just a data structure where we put the right parts in the right place. But what exacly do we fill in for our register values?

## Registers

Registers are small places to store values. Every instruction set will have a different number of these registers, different sizes of registers, different kinds of registers, and different naming conventions for registers. For aarch64, there are 31 general purpose registers numbered X0 through X30 for  64-bit registers. Let's say we want to add 42 to register X0 and store the result in X1 we'd use this binary number.

<table
   <thead>
      <tr>
         <td>sf</td>
         <td colspan="8">operation</td>
         <td>sh</td>
         <td colspan="12">imm12</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rd</td>
      </tr>
   </thead>
   <tbody>
			<tr>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
      </tr>
   </tbody>
</table>
To encode our registers into our instruction, we just use their number. So register X0 would be 00000 and register X18 would be `10010`. Registers are simply places we can store values. But by convention different registers can be used for different things, in a little bit we explore this concept in more detail. For right now just consider all registers to be more or less the same. 

Writing out all these binary numbers all the time (or even converting them to hex) can often be tedious. So instead we usually talk about instructions in a simple text format called assembly.

```assembly
add	x1, x0, #0x2a 
```

In order to feel cool, people usually write numbers in assembly as hex values. This is just the number 42. You can see that assembly hide some detail of the encoding we just made. We don't think about sf, sh, what size our number is, that a register is Rn vs Rd. Instead, the destination comes first and the arguments after. Because of this lack of detail, a single assembly instruction `add` might actually map to many different machine code instructions depending on its arguments.

## Memory

The last piece we have to understand for machine code is memory. To understand what is going on with memory, we will look at an instruction that lets us store things in memory. This the instruction called "STR" but we are just going to call it store.

// Change to https://developer.arm.com/documentation/ddi0602/2022-06/Base-Instructions/STUR--Store-Register--unscaled--

<table
   <thead>
      <tr>
         <td>31</td>
         <td>30</td>
         <td>29</td>
         <td>28</td>
         <td>27</td>
         <td>26</td>
         <td>25</td>
         <td>24</td>
         <td>23</td>
         <td>22</td>
         <td>21</td>
         <td>20</td>
         <td>19</td>
         <td>18</td>
         <td>17</td>
         <td>16</td>
         <td>15</td>
         <td>14</td>
         <td>13</td>
         <td>12</td>
         <td>11</td>
         <td>10</td>
         <td>9</td>
         <td>8</td>
         <td>7</td>
         <td>6</td>
         <td>5</td>
         <td>4</td>
         <td>3</td>
         <td>2</td>
         <td>1</td>
         <td>0</td>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>1</td>
         <td>x</td>
         <td>1</td>
         <td>1</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td colspan="5">Rm</td>
         <td colspan="3">option</td>
         <td>S</td>
         <td>1</td>
         <td>0</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rt</td>
      </tr>
   </tbody>
</table>

This instruction is a bit more complicated so let's break down what it is doing. Store lets us put the value of some register (Rt) into memory. Our address is stored in Rm and we can provide an offset in Rt. Let's say we want to store X2 at the address stored in X0 with an offset stored in X1. We can write  this in assembly like so:

```assembly
str X2, [X0, X1]
```

But what does this even mean? Why would we use this. Imagine that X0 holds the address of an array. This code is equivalent to 

```javascript
// X0 = my_array
// X1 = i
// X2 = my_variable
my_array[i] = my_variable;
```

