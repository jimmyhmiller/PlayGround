const fs = require('fs');
const path = require('path');

// Use the OpenAI API key from the environment variable
const apiKey = process.env.OPENAI_KEY;

// Function to simplify the name
const simplifyName = (name) => {
    return name.toLowerCase().replace(/\W+/g, '_');
};

// Function to generate image using OpenAI API
const generateImage = async (prompt) => {
    const response = await fetch('https://api.openai.com/v1/images/generations', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            prompt: prompt,
            n: 1,
            size: "1792x1024",
            model: "dall-e-3"
        })
    });

    const data = await response.json();
    if (response.ok) {
        return data.data[0].url;
    } else {
        throw new Error(data.error.message);
    }
};

// Main function to read JSON from stdin, generate images, and save them
const generateImagesFromJson = async () => {
    let jsonData = '';

    // Read input from stdin
    process.stdin.on('data', (chunk) => {
        jsonData += chunk;
    });

    process.stdin.on('end', async () => {
        const entries = JSON.parse(jsonData);
        
        for (const entry of entries) {
            const { name, description } = entry;
            const simplifiedName = simplifyName(name);
            try {
                const imageUrl = await generateImage(description);
                const imageResponse = await fetch(imageUrl);
                const arrayBuffer = await imageResponse.arrayBuffer();
                const buffer = Buffer.from(arrayBuffer);
                const imagePath = path.join(__dirname, `${simplifiedName}.png`);
                fs.writeFileSync(imagePath, buffer);
                console.log(`Image saved as ${simplifiedName}.png`);
            } catch (error) {
                console.error(`Failed to generate image for ${name}:`, error);
            }
        }
    });
};

// Run the main function
generateImagesFromJson();