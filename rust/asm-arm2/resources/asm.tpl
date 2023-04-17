{% comment %} 
{%- for instruction in instructions %}

// {{ instruction.title }}
// {{ instruction.description }}
{%- for comment in instruction.argument_comments %}
// {{comment}}
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

enum Asm {
{%- for instruction in instructions %}
    {{instruction.name}} {
        {%- for field in instruction.fields -%}
        {%- if field.required %}
        {{ field.name }}: u32,
        {%- endif -%}
        {%- endfor %}
    },
{%- endfor %}
}