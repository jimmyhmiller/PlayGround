{% comment %} 
{%- for instruction in instructions %}

// {{ instruction.title }}
// {{ instruction.description }}
{%- for arg_comment in instruction.argument_comments %}
// {{arg_comment}}
{%- endfor %}


struct {{instruction.name}} {
    {%- for field in instruction.fields %}
    {%- if field.required %}
    {{ field.name }}: u32,
    {%- endif -%}
    {%- endfor %}
}

{%- endfor %}

{% endcomment %}

#[derive(Debug)]
enum Size {
    S32,
    S64,
}

#[derive(Debug)]
struct Register {
    size: Size,
    index: u8,
}

impl Register {
    fn encode(&self) -> u8 {
        self.index
    }
}

impl Shl<u32> for &Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}


#[derive(Debug)]
enum Asm {
{%- for instruction in instructions %}
    // {{ instruction.title }}
    // {{ instruction.description }}
    {%- for arg_comment in instruction.argument_comments %}
    // {{arg_comment}}
    {%- endfor %}
    {{instruction.name}} {
        {%- for field in instruction.fields -%}
        {%- if field.required %}
        {{ field.name }}: {% if field.kind == "Register" %}Register{% else %}u32{%- endif -%},
        {%- endif -%}
        {%- endfor %}
    },
{%- endfor %}
}

impl Asm {
    fn encode(&self) -> u32 {
        match self {
            {%- for instruction in instructions %}
            Asm::{{instruction.name}}{ 
                {%- for field in instruction.fields -%}  
                    {%- if field.required %} 
                        {{ field.name }}, 
                    {%- endif -%}
                {% endfor %}
            } => {
            0b{%- for field in instruction.fields -%}{{field.bits}}_{% endfor %}
            {%- for field in instruction.fields -%}
                {%- if field.required %} 
                | {{field.name}} << {{field.shift}}
                {%- endif -%}

            {%- endfor%}
            }
            {% endfor %}
        }
    }
}