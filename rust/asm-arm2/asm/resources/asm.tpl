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
    pub fn sf(&self) -> i32 {
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


pub fn truncate_imm<T: Into<i32>, const WIDTH: usize>(imm: T) -> u32 {
    let value: i32 = imm.into();
    let masked = (value as u32) & ((1 << WIDTH) - 1);

    // Assert that we didn't drop any bits by truncating.
    if value >= 0 {
        assert_eq!(value as u32, masked);
    } else {
        assert_eq!(value as u32, masked | (u32::MAX << WIDTH));
    }

    masked
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
        {{ field.name }}: {% if field.kind == "Register" %}Register{% else %}i32{%- endif -%},
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
                |     
                    {%- if field.name == "imm19" -%}
                       truncate_imm::<_, 19>(*{{field.name}}) << {{field.shift}}
                    {%- elsif field.name == "imm26" -%}
                       truncate_imm::<_, 26>(*{{field.name}}) << {{field.shift}}
                    {%- else %}
                        {% if field.kind == "Register" %}
                            {{field.name}} << {{field.shift}}
                        {% else %}
                            (*{{field.name}} as u32) << {{field.shift}}
                        {% endif %}
                    {%- endif -%}
 
                {%- endif -%}

            {%- endfor%}
            }
            {% endfor %}
        }
    }
}