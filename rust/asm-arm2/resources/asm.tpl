use std::ops::Shl;

#[derive(Debug)]
pub enum Size {
    S32,
    S64,
}

#[derive(Debug)]
pub struct Register {
    pub size: Size,
    pub index: u8,
}

impl Register {
    pub fn sf(&self) -> u32 {
        match self.size {
            Size::S32 => 0,
            Size::S64 => 1,
        }
    }
}

impl Register {
    pub fn encode(&self) -> u8 {
        self.index
    }
}

{% comment %}This is inclusive{% endcomment %}
{% for i in (0..30) %}
pub const X{{i}}: Register = Register {
    index: {{i}},
    size: Size::S64,
};
{% endfor %}

pub const XZR: Register = Register {
    index: 31,
    size: Size::S64,
};

impl Shl<u32> for &Register {
    type Output = u32;

    fn shl(self, rhs: u32) -> Self::Output {
        (self.encode() as u32) << rhs
    }
}


#[derive(Debug)]
pub enum Asm {
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
    pub fn encode(&self) -> u32 {
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