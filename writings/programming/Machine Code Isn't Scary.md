# Machine Code Isn't Scary

The first programming language I ever learned was ActionScript. Writing code for Macromedia's Flash might be the furthest away from "bare metal" as you can possibly get. As I continued learning new languages, this starting heritage stuck with me. I was mostly interested in high-level, "web languages". Low-level languages felt impenetrable. Over time, I learned a bit more about them here and there, but for some reason, this notion stuck with me. Low-level things are scary, and machine code epitomized that most directly. When I Googled things asking about writing in straight machine code, I was met with discouraging messages rather than learning. 

Eventually, I decided I needed to overcome this belief if I was going to achieve my goals. In doing so, I learned something I didn't expect.

> Machine code isn't scary. If you can make sure your JSON conforms to a JSON schema, you can write machine code.

### Which Machine Code?

One problem with machine code is that there isn't simply one standard. There are many different "instruction sets" depending on the processor. Most modern PCs use x86-64 machine code, but newer Macs, Raspberry Pis, and most mobile devices use ARM. There are other architectures out there, especially as you go back in time. The goal of this article won't be to give you a deep understanding of any particular instruction set, but instead, to give you enough information about how machine code typically works so you cannot be afraid of machine code. So we will start by having our examples be in ARM 64-bit (also written as aarch64). Once we have a decent understanding of that, we will talk a bit about x86-64.

## Machine Code Basics

To understand the basics of machine code, you need three concepts:

1. Instructions
2. Registers
3. Memory

Instructions are exactly what they sound like; they are the code that will run. Machine code instructions are just numbers. In fact, in AArch64, every instruction is a 32-bit number. Instructions encode what operation the machine should run (add, move, subtract, jump, etc.) and accept some arguments for what data to operate on. These arguments might be constants (meaning like the number 2; these constants are often called "immediates"), but they can also be registers or a memory address. For now, just think of a register as a variable and memory as a list.

### Arm Instructions

Here is an example of the instruction `add immediate`.

<table>
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

Now this might look a bit confusing, but once you've seen these tables long enough, they start to be fairly straightforward. Each column in this table represents a single bit in a 32-bit number. If the value is a 0 or 1, that just means it is already filled in. If it has a label, it is a variable that needs to be filled in. `sf` tells us whether the registers we are going to use are 64-bit or 32-bit registers. `sh` stands for shift. `sh` goes in conjunction with imm12, which stands for a 12-bit immediate (constant).  So if we want to add `42` to something, we would put `000000101010` in for `imm12` and set sh to 0 (meaning we aren't shifting the number). But what if we want to represent a number larger than 12 bits? Well, the add instruction doesn't let us represent all such numbers; setting sh to 1 lets us shift our number by 12. So for example we can represent `172032172032` by leaving our 42 alone and setting sh to 1. This is a clever technique for encoding larger numbers in a small space. Finally, to Rn and Rd. R variables mean that these are registers. Rn is our argument to add, and Rd is our destination.

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

Our add instruction is really just a data structure where we put the right parts in the right place. But what exactly fills in for our register values?

## Registers

Registers are small places to store values. Every instruction set will have a different number of these registers, different sizes of registers, different kinds of registers, and different naming conventions for registers. For AArch64, there are 31 general-purpose registers numbered X0 through X30 for  64-bit registers. Let's say we want to add 42 to register X0 and store the result in X1; we use this binary number.

<table><tablehead>
      <tr>
         <td>sf</td>
         <td colspan="8">operation</td>
         <td>sh</td>
         <td colspan="12">imm12</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rd</td>
      </tr>
</table>
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
To encode our registers into our instruction, we just use their number. So register X0 would be 00000 and register X18 would be `10010`. Registers are simply places where we can store values. But by convention, registers can be used for different things. These are called calling conventions and they are how "higher" level languages like C encode function calls.

Writing out all these binary numbers all the time (or even converting them to hex) can often be tedious. So instead, we usually talk about instructions in a simple text format called assembly.

```assembly
add   x1, x0, #0x2a 
```

In order to feel cool, people usually write numbers in assembly as hex values. This is just the number 42. You can see that assembly hides some of the details of the encoding we just made. We don't think about sf, sh, what size our number is, that a register is Rn vs Rd. Instead, the destination comes first and the arguments after. Because of this lack of detail, a single assembly instruction `add` might actually map to many different machine code instructions depending on its arguments.

## Memory

The last piece we have to understand for machine code is memory. To understand what is going on with memory, we will look at an instruction that lets us store things in memory. This instruction is called "STR" but we are just going to call it store.

<table>
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
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td colspan="12">imm12</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rt</td>
      </tr>
   </tbody>
</table>
Using this instruction, we are going to store some value (RT) into the address (RN) + some offset (imm12). So if we think about memory as a big array, this instruction is like writing into that array. `array[offset] = value`. The x here is like our sf before, it controls whether we are using 64-bit values or not. If we want to make this concrete, let's say we have a value in X2, we have an address of memory in X1 and we want to store a value 2 bytes offset from that. We would get this structure:

<table>
   <thead>
      <tr>
         <td></td>
         <td>x</td>
         <td colspan="8">operation</td>
         <td colspan="12">imm12</td>
         <td colspan="5">Rn</td>
         <td colspan="5">Rt</td>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>1</td>
         <td>1</td>
         <!-- operation -->
         <td>1</td>
         <td>1</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
         <td>0</td>
         <!-- Imm12 -->
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
         <td>0</td>
         <!-- RN -->
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
        <!-- RT -->
         <td>0</td>
         <td>0</td>
         <td>0</td>
         <td>1</td>
         <td>0</td>
      </tr>
   </tbody>
</table>
Since writing that all is tedious, we often just write the assembly notation. We are storing the value in x2 based on the address stored in x1 + 2.

```assembly
str x2, [x1, #0x2]
```

## X86-64

X86 encoding is a bit different, but it more or less has the same parts. We are still working with instructions, registers, and memory. Some names are a bit different. Instead of the consistent 0-30 naming, we get the historical baggage of the following 64-bit bits: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, arsp8-r15). However, the biggest difference is that x86 is not a fixed-fixed width instruction set. We can't simply give a nice little diagram of every instruction using 32 bits. Instead, instructions are assembled into parts. These parts are given different names; when you see an instruction encoding, it tells you how to put the parts together.

### REX

<table style="width: auto; border-collapse: collapse; text-align: center;">
  <tr>
    <th>7</th><th>6</th><th>5</th><th>4</th><th>3</th><th>2</th><th>1</th><th>0</th>
  </tr>
  <tr>
    <td>0</td><td>1</td><td>0</td><td>0</td><td>W</td><td>R</td><td>X</td><td>B</td>
  </tr>
</table>

The first part is called the REX. This is a prefix that we can use to help us with 64-bit operations. Not sure if there is an official justification for the name REX, but my understanding is that it is the "Register Extension Prefix". Unfortunately, because the REX is a prefix, it will only make sense when we see what comes later. REX is there for backward compatibility. The W in REX lets us signal that we are using 64-bit or not for certain operations. The R, B, and B will "extend" our registers in certain operations. In other words, it allows more registers than you used to be able to (These are those r8-r15 registers with a different naming convention than the older registers). We need these because, before 64-bit x86, we had fewer registers and our instructions only had 3 bits per register. With 16 registers, we need an extra bit.

### ModR/M

<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse; text-align: center; width: auto;">
  <thead>
    <tr>
      <td style="width: 30px;">7</td>
      <td style="width: 30px;">6</td>
      <td style="width: 30px;">5</td>
      <td style="width: 30px;">4</td>
      <td style="width: 30px;">3</td>
      <td style="width: 30px;">2</td>
      <td style="width: 30px;">1</td>
      <td style="width: 30px;">0</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">mod</td>
      <td colspan="3">reg</td>
      <td colspan="3">rm</td>
    </tr>
  </tbody>
</table>
ModR/M keeps up with the tradition of naming things incredibly short and confusing names. `mod` actually means Mode. `mod` tells us if `rm` is acting as a register or if it is a pointer to memory. If `mod == 11` then rm is being used as a register, otherwise, it is being used as a pointer. `reg` just is a register.

### OpCode

`OpCode` is simple, it is a number. It can be 1-3 bytes long.

## Putting It Together

There are other parts, but we won't cover them here. With just these parts, we can build up an instruction. Let's say we want to move a 32-bit register to a 64-bit register. We can consult [a table of instruction encodings](https://www.felixcloutier.com/x86/mov) and we will get this:

```
REX.W + C7 /0 id
```

So now we can assemble our parts and make our instruction. Let's start with REX.W. This notation just means REX with W set to 1. Then there’s B8, which is just a number written in hex. `/0` is yet more shorthand for using the ModR/M but setting the reg to 0. Finally, `id` means "immediate doubleword", in other words, a constant number that is 32 bits long. So given all that, we can write our instruction. So let's move the number 42 to the rbx register.

| Byte Index | Bits  | Description                                  |
| ---------- | ----- | -------------------------------------------- |
| Byte 0     | 55–48 | 01001000      REX.W = 1                      |
| Byte 1     | 47–40 | 11000111      Opcode C7                      |
| Byte 2     | 39–32 | 11000011      ModR/M: reg=000, r/m=011 (RBX) |
| Byte 3     | 31–24 | 00101010      42                             |
| Byte 4     | 23–16 | 00000000      the rest of 42                 |
| Byte 5     | 15–8  | 00000000                                     |
| Byte 6     | 7–0   | 00000000                                     |

Why is RBX 011? Well, because [the table](https://wiki.osdev.org/X86-64_Instruction_Encoding#Registers) says so. Yeah, I did say that x86 is a bit weird.

## The Rest of It

I won't pretend that this is all you need. But I will say that starting here can get you further than you think. There are some other things to learn, like various flags for things like overflow, there’s also calling conventions, which are about which registers you use when for things like function calls. We haven't really talked about the stack here, but that's memory that you write to to keep track of things. Nor have we talked about jumps, or how to encode larger immediates in ARM, but you’ve gotten the basics. It’s easier than you would think to hop on [compiler explorer](https://godbolt.org/) and learn how things are done.

Learning machine code and writing things at this low level has unlocked so many things that were mental blocks for me before. Relying on libraries made by others to do these low-level things always left a gap in my knowledge that made me doubt my understanding. Even if I intellectually could explain things, actually doing has made a huge difference for me. So if you, like me, find low-level things intimidating, I can't recommend enough starting from scratch, at the lowest level you can.
