// Calculate the exact byte offsets for each field in the Quad struct
// to see if our layout matches WGSL's expectations

function calculateLayout() {
    let offset = 0;
    const fields = [];

    function addField(name, size, alignment = 4) {
        // Align offset
        offset = Math.ceil(offset / alignment) * alignment;
        fields.push({ name, offset, size });
        offset += size;
    }

    addField('order', 4);
    addField('border_style', 4);
    addField('bounds', 16);
    addField('content_mask', 16);

    // Background has 16-byte alignment
    offset = Math.ceil(offset / 16) * 16;
    fields.push({ name: 'PADDING_BEFORE_BACKGROUND', offset, size: 0 });

    // Background struct
    addField('background.tag', 4);
    addField('background.color_space', 4);
    addField('background.solid', 16);
    addField('background.gradient_angle', 4);

    // Colors array needs 16-byte alignment
    offset = Math.ceil(offset / 16) * 16;
    fields.push({ name: 'PADDING_BEFORE_COLORS', offset, size: 0 });

    // colors[0]
    const colors0Start = offset;
    addField('background.colors[0].color', 16);
    addField('background.colors[0].percentage', 4);
    // Pad to 32-byte stride from start of colors[0]
    const colors0End = offset;
    const colors1Start = colors0Start + 32;
    if (offset < colors1Start) {
        fields.push({ name: 'PADDING_AFTER_COLORS[0]', offset, size: colors1Start - offset });
        offset = colors1Start;
    }

    // colors[1]
    addField('background.colors[1].color', 16);
    addField('background.colors[1].percentage', 4);
    // Pad to 32-byte stride from start of colors[1]
    const colors1End = offset;
    const paddedEnd = colors1Start + 32;
    if (offset < paddedEnd) {
        fields.push({ name: 'PADDING_AFTER_COLORS[1]', offset, size: paddedEnd - offset });
        offset = paddedEnd;
    }

    addField('background.pad', 4);
    // Round Background to 16-byte multiple
    const backgroundEnd = offset;
    offset = Math.ceil(offset / 16) * 16;
    if (offset > backgroundEnd) {
        fields.push({ name: 'PADDING_AFTER_BACKGROUND_PAD', offset: backgroundEnd, size: offset - backgroundEnd });
    }

    addField('border_color', 16);
    addField('corner_radii', 16);
    addField('border_widths', 16);
    addField('transform', 32);
    addField('opacity', 4);
    addField('pad', 4);

    console.log('\n=== Quad Struct Layout ===\n');
    fields.forEach(f => {
        const floatOffset = f.offset / 4;
        console.log(`${f.name.padEnd(40)} byte ${String(f.offset).padStart(3)} (float ${String(floatOffset).padStart(3)})  size ${f.size}`);
    });

    console.log(`\nTotal size: ${offset} bytes = ${offset / 4} floats`);
}

calculateLayout();
