import { Scene, Quad, Shadow, Background, Underline, MonochromeSprite, PolychromeSprite, Path } from './core/primitives.js';
import { Bounds, Point, Size, Corners, Edges, Hsla, Transform } from './core/geometry.js';
import { PathBuilder, parseSVGPath } from './core/path.js';
import { WebGPURenderer } from './renderer/webgpu-renderer.js';
import { TextRenderer } from './utils/text-renderer.js';
import { ImageLoader, ProceduralImages } from './utils/image-loader.js';
import { HitTester } from './utils/hit-tester.js';
import { TextMeasurer, getAlignedTextPosition, TextAlign, VerticalAlign, truncateText } from './utils/text-layout.js';
import { Profiler, MemoryMonitor } from './utils/profiler.js';
import {
    generateCircleGlyph,
    generateStarGlyph,
    generateHeartGlyph,
    generateGradientPattern,
    generateCheckerboard,
    generateCircularGradient
} from './utils/texture-gen.js';

async function main() {
    const canvas = document.getElementById('canvas');
    const info = document.getElementById('info');
    const errorDiv = document.getElementById('error');

    try {
        info.textContent = 'Initializing WebGPU...';

        // Create renderer
        const renderer = new WebGPURenderer(canvas);
        await renderer.initialize();

        info.textContent = 'Generating textures...';

        // Generate and upload monochrome glyphs
        const circleData = generateCircleGlyph(64);
        const circleTile = renderer.monochromeAtlas.getOrInsert('circle', 64, 64, circleData);

        const starData = generateStarGlyph(64);
        const starTile = renderer.monochromeAtlas.getOrInsert('star', 64, 64, starData);

        const heartData = generateHeartGlyph(64);
        const heartTile = renderer.monochromeAtlas.getOrInsert('heart', 64, 64, heartData);

        // Generate and upload polychrome patterns
        const gradientData = generateGradientPattern(128, 128);
        const gradientTile = renderer.polychromeAtlas.getOrInsert('gradient', 128, 128, gradientData);

        const checkerData = generateCheckerboard(64, 64, 8);
        const checkerTile = renderer.polychromeAtlas.getOrInsert('checker', 64, 64, checkerData);

        const circularData = generateCircularGradient(128);
        const circularTile = renderer.polychromeAtlas.getOrInsert('circular', 128, 128, circularData);

        // Create text renderer
        const textRenderer = new TextRenderer(renderer);

        // Create image loader
        const imageLoader = new ImageLoader(renderer);

        info.textContent = 'Creating scene...';

        // Create a test scene
        const scene = new Scene();

        const fullMask = new Bounds(
            new Point(0, 0),
            new Size(canvas.width, canvas.height)
        );

        // === ROW 1: GRADIENTS WITH BORDERS ===

        // Gradient quad 1
        const shadow1 = new Shadow();
        shadow1.bounds = new Bounds(new Point(25, 25), new Size(140, 90));
        shadow1.cornerRadii = Corners.uniform(12);
        shadow1.blurRadius = 6;
        shadow1.color = Hsla.black(0.3);
        shadow1.contentMask.bounds = fullMask;
        scene.insertShadow(shadow1);

        const gradQuad1 = new Quad();
        gradQuad1.bounds = new Bounds(new Point(20, 20), new Size(150, 100));
        gradQuad1.background = Background.LinearGradient(45, [
            { color: Hsla.rgb(0.2, 0.5, 0.9, 1), position: 0 },
            { color: Hsla.rgb(0.8, 0.3, 0.6, 1), position: 1 }
        ], 0);
        gradQuad1.cornerRadii = Corners.uniform(15);
        gradQuad1.borderWidths = Edges.uniform(3);
        gradQuad1.borderColor = Hsla.rgb(0.1, 0.2, 0.4, 1);
        gradQuad1.contentMask.bounds = fullMask;
        scene.insertQuad(gradQuad1);

        // Gradient quad 2 (Oklab)
        const shadow2 = new Shadow();
        shadow2.bounds = new Bounds(new Point(195, 25), new Size(140, 90));
        shadow2.cornerRadii = Corners.uniform(15);
        shadow2.blurRadius = 8;
        shadow2.color = Hsla.black(0.25);
        shadow2.contentMask.bounds = fullMask;
        scene.insertShadow(shadow2);

        const gradQuad2 = new Quad();
        gradQuad2.bounds = new Bounds(new Point(190, 20), new Size(150, 100));
        gradQuad2.background = Background.LinearGradient(90, [
            { color: Hsla.rgb(0.95, 0.4, 0.3, 1), position: 0 },
            { color: Hsla.rgb(0.3, 0.9, 0.5, 1), position: 1 }
        ], 1); // Oklab
        gradQuad2.cornerRadii = Corners.uniform(18);
        gradQuad2.borderWidths = new Edges(4, 6, 4, 6);
        gradQuad2.borderColor = Hsla.white(0.9);
        gradQuad2.contentMask.bounds = fullMask;
        scene.insertQuad(gradQuad2);

        // === ROW 2: SPRITES ===

        // Monochrome sprites (glyphs)
        const circleSprite = new MonochromeSprite();
        circleSprite.bounds = new Bounds(new Point(40, 150), new Size(48, 48));
        circleSprite.color = Hsla.rgb(0.9, 0.2, 0.3, 1);
        circleSprite.tile = circleTile;
        circleSprite.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(circleSprite);

        const starSprite = new MonochromeSprite();
        starSprite.bounds = new Bounds(new Point(110, 150), new Size(48, 48));
        starSprite.color = Hsla.rgb(0.9, 0.8, 0.2, 1);
        starSprite.tile = starTile;
        starSprite.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(starSprite);

        const heartSprite = new MonochromeSprite();
        heartSprite.bounds = new Bounds(new Point(180, 150), new Size(48, 48));
        heartSprite.color = Hsla.rgb(0.95, 0.3, 0.5, 1);
        heartSprite.tile = heartTile;
        heartSprite.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(heartSprite);

        // Polychrome sprites (images/patterns)
        const gradientSprite = new PolychromeSprite();
        gradientSprite.bounds = new Bounds(new Point(250, 140), new Size(80, 80));
        gradientSprite.tile = gradientTile;
        gradientSprite.cornerRadii = Corners.uniform(10);
        gradientSprite.opacity = 1.0;
        gradientSprite.contentMask.bounds = fullMask;
        scene.insertPolychromeSprite(gradientSprite);

        // Grayscale polychrome sprite
        const grayscaleSprite = new PolychromeSprite();
        grayscaleSprite.bounds = new Bounds(new Point(340, 140), new Size(80, 80));
        grayscaleSprite.tile = circularTile;
        grayscaleSprite.cornerRadii = Corners.uniform(10);
        grayscaleSprite.opacity = 1.0;
        grayscaleSprite.grayscale = true;
        grayscaleSprite.contentMask.bounds = fullMask;
        scene.insertPolychromeSprite(grayscaleSprite);

        // Low opacity polychrome sprite
        const fadedSprite = new PolychromeSprite();
        fadedSprite.bounds = new Bounds(new Point(430, 140), new Size(80, 80));
        fadedSprite.tile = checkerTile;
        fadedSprite.cornerRadii = Corners.uniform(10);
        fadedSprite.opacity = 0.5;
        fadedSprite.contentMask.bounds = fullMask;
        scene.insertPolychromeSprite(fadedSprite);

        // === ROW 3: UNDERLINES ===

        // Straight underlines
        const straightUnderline1 = new Underline();
        straightUnderline1.bounds = new Bounds(new Point(20, 250), new Size(150, 10));
        straightUnderline1.color = Hsla.rgb(0.2, 0.4, 0.8, 1);
        straightUnderline1.thickness = 3;
        straightUnderline1.wavy = 0;
        straightUnderline1.contentMask.bounds = fullMask;
        scene.insertUnderline(straightUnderline1);

        const straightUnderline2 = new Underline();
        straightUnderline2.bounds = new Bounds(new Point(20, 275), new Size(150, 10));
        straightUnderline2.color = Hsla.rgb(0.8, 0.3, 0.4, 1);
        straightUnderline2.thickness = 5;
        straightUnderline2.wavy = 0;
        straightUnderline2.contentMask.bounds = fullMask;
        scene.insertUnderline(straightUnderline2);

        // Wavy underlines
        const wavyUnderline1 = new Underline();
        wavyUnderline1.bounds = new Bounds(new Point(190, 250), new Size(150, 20));
        wavyUnderline1.color = Hsla.rgb(0.3, 0.8, 0.5, 1);
        wavyUnderline1.thickness = 4;
        wavyUnderline1.wavy = 1;
        wavyUnderline1.contentMask.bounds = fullMask;
        scene.insertUnderline(wavyUnderline1);

        const wavyUnderline2 = new Underline();
        wavyUnderline2.bounds = new Bounds(new Point(190, 280), new Size(150, 20));
        wavyUnderline2.color = Hsla.rgb(0.9, 0.6, 0.2, 1);
        wavyUnderline2.thickness = 6;
        wavyUnderline2.wavy = 1;
        wavyUnderline2.contentMask.bounds = fullMask;
        scene.insertUnderline(wavyUnderline2);

        // === ROW 4: MIXED FEATURES ===

        // Card with shadow, gradient, border, and sprites
        const cardShadow = new Shadow();
        cardShadow.bounds = new Bounds(new Point(25, 325), new Size(300, 120));
        cardShadow.cornerRadii = Corners.uniform(20);
        cardShadow.blurRadius = 15;
        cardShadow.color = Hsla.black(0.4);
        cardShadow.contentMask.bounds = fullMask;
        scene.insertShadow(cardShadow);

        const card = new Quad();
        card.bounds = new Bounds(new Point(20, 320), new Size(310, 130));
        card.background = Background.LinearGradient(135, [
            { color: Hsla.rgb(0.95, 0.95, 0.98, 1), position: 0 },
            { color: Hsla.rgb(0.85, 0.88, 0.95, 1), position: 1 }
        ], 0);
        card.cornerRadii = Corners.uniform(20);
        card.borderWidths = Edges.uniform(2);
        card.borderColor = Hsla.rgb(0.7, 0.75, 0.85, 1);
        card.contentMask.bounds = fullMask;
        scene.insertQuad(card);

        // Sprites on the card
        const cardCircle = new MonochromeSprite();
        cardCircle.bounds = new Bounds(new Point(40, 345), new Size(40, 40));
        cardCircle.color = Hsla.rgb(0.3, 0.5, 0.9, 1);
        cardCircle.tile = circleTile;
        cardCircle.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(cardCircle);

        const cardStar = new MonochromeSprite();
        cardStar.bounds = new Bounds(new Point(100, 345), new Size(40, 40));
        cardStar.color = Hsla.rgb(0.9, 0.7, 0.2, 1);
        cardStar.tile = starTile;
        cardStar.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(cardStar);

        const cardHeart = new MonochromeSprite();
        cardHeart.bounds = new Bounds(new Point(160, 345), new Size(40, 40));
        cardHeart.color = Hsla.rgb(0.95, 0.3, 0.4, 1);
        cardHeart.tile = heartTile;
        cardHeart.contentMask.bounds = fullMask;
        scene.insertMonochromeSprite(cardHeart);

        // Polychrome pattern
        const cardPattern = new PolychromeSprite();
        cardPattern.bounds = new Bounds(new Point(230, 335), new Size(80, 80));
        cardPattern.tile = circularTile;
        cardPattern.cornerRadii = Corners.uniform(12);
        cardPattern.opacity = 0.9;
        cardPattern.contentMask.bounds = fullMask;
        scene.insertPolychromeSprite(cardPattern);

        // === ROW 5: PATTERN BACKGROUNDS ===

        // Pattern 1: Diagonal stripes 45 degrees
        const pattern1 = new Quad();
        pattern1.bounds = new Bounds(new Point(20, 470), new Size(100, 80));
        pattern1.background = Background.Pattern(45,
            Hsla.rgb(0.2, 0.4, 0.8, 1),
            Hsla.rgb(0.8, 0.9, 0.95, 1),
            10);
        pattern1.cornerRadii = Corners.uniform(10);
        pattern1.contentMask.bounds = fullMask;
        scene.insertQuad(pattern1);

        // Pattern 2: Vertical stripes
        const pattern2 = new Quad();
        pattern2.bounds = new Bounds(new Point(140, 470), new Size(100, 80));
        pattern2.background = Background.Pattern(0,
            Hsla.rgb(0.9, 0.3, 0.4, 1),
            Hsla.rgb(1.0, 0.9, 0.9, 1),
            8);
        pattern2.cornerRadii = Corners.uniform(10);
        pattern2.contentMask.bounds = fullMask;
        scene.insertQuad(pattern2);

        // Pattern 3: Fine diagonal stripes
        const pattern3 = new Quad();
        pattern3.bounds = new Bounds(new Point(260, 470), new Size(100, 80));
        pattern3.background = Background.Pattern(135,
            Hsla.rgb(0.3, 0.8, 0.5, 1),
            Hsla.rgb(0.9, 0.98, 0.95, 1),
            6,
            0); // Stripe pattern
        pattern3.cornerRadii = Corners.uniform(10);
        pattern3.borderWidths = Edges.uniform(2);
        pattern3.borderColor = Hsla.rgb(0.2, 0.6, 0.3, 1);
        pattern3.contentMask.bounds = fullMask;
        scene.insertQuad(pattern3);

        // === ROW 6: DOT PATTERNS ===

        // Dot pattern 1: Blue dots
        const dotPattern1 = new Quad();
        dotPattern1.bounds = new Bounds(new Point(20, 565), new Size(100, 30));
        dotPattern1.background = Background.Pattern(0,
            Hsla.rgb(0.95, 0.95, 0.98, 1),
            Hsla.rgb(0.2, 0.4, 0.9, 1),
            12,
            1); // Dot pattern
        dotPattern1.cornerRadii = Corners.uniform(5);
        dotPattern1.contentMask.bounds = fullMask;
        scene.insertQuad(dotPattern1);

        // Dot pattern 2: Red dots
        const dotPattern2 = new Quad();
        dotPattern2.bounds = new Bounds(new Point(140, 565), new Size(100, 30));
        dotPattern2.background = Background.Pattern(0,
            Hsla.rgb(0.98, 0.95, 0.95, 1),
            Hsla.rgb(0.9, 0.2, 0.3, 1),
            10,
            1); // Dot pattern
        dotPattern2.cornerRadii = Corners.uniform(5);
        dotPattern2.contentMask.bounds = fullMask;
        scene.insertQuad(dotPattern2);

        // Dot pattern 3: Green dots with border
        const dotPattern3 = new Quad();
        dotPattern3.bounds = new Bounds(new Point(260, 565), new Size(100, 30));
        dotPattern3.background = Background.Pattern(0,
            Hsla.rgb(0.95, 0.98, 0.95, 1),
            Hsla.rgb(0.3, 0.8, 0.4, 1),
            8,
            1); // Dot pattern
        dotPattern3.cornerRadii = Corners.uniform(5);
        dotPattern3.borderWidths = Edges.uniform(1);
        dotPattern3.borderColor = Hsla.rgb(0.2, 0.6, 0.3, 1);
        dotPattern3.contentMask.bounds = fullMask;
        scene.insertQuad(dotPattern3);

        // === ROW 7: RADIAL GRADIENTS ===

        // Radial gradient 1: Center circle, sRGB
        const radial1 = new Quad();
        radial1.bounds = new Bounds(new Point(20, 610), new Size(100, 80));
        radial1.background = Background.RadialGradient(
            0.5, 0.5, 0.6, // center (0.5, 0.5) = center, radius 0.6
            [
                { color: Hsla.rgb(1.0, 0.9, 0.3, 1), position: 0 },
                { color: Hsla.rgb(0.9, 0.3, 0.5, 1), position: 1 }
            ],
            0 // sRGB color space
        );
        radial1.cornerRadii = Corners.uniform(10);
        radial1.contentMask.bounds = fullMask;
        scene.insertQuad(radial1);

        // Radial gradient 2: Offset center, Oklab
        const radial2 = new Quad();
        radial2.bounds = new Bounds(new Point(140, 610), new Size(100, 80));
        radial2.background = Background.RadialGradient(
            0.3, 0.3, 0.8, // center offset to top-left
            [
                { color: Hsla.rgb(0.3, 0.8, 0.9, 1), position: 0 },
                { color: Hsla.rgb(0.5, 0.3, 0.8, 1), position: 1 }
            ],
            1 // Oklab color space
        );
        radial2.cornerRadii = Corners.uniform(10);
        radial2.contentMask.bounds = fullMask;
        scene.insertQuad(radial2);

        // Radial gradient 3: Large radius with border
        const radial3 = new Quad();
        radial3.bounds = new Bounds(new Point(260, 610), new Size(100, 80));
        radial3.background = Background.RadialGradient(
            0.5, 0.5, 1.2, // radius extends beyond bounds
            [
                { color: Hsla.rgb(0.9, 0.4, 0.3, 1), position: 0 },
                { color: Hsla.rgb(0.3, 0.4, 0.9, 1), position: 1 }
            ],
            1 // Oklab
        );
        radial3.cornerRadii = Corners.uniform(10);
        radial3.borderWidths = Edges.uniform(2);
        radial3.borderColor = Hsla.rgb(0.2, 0.2, 0.3, 1);
        radial3.contentMask.bounds = fullMask;
        scene.insertQuad(radial3);

        // === ROW 8: ADDITIONAL PATTERNS (CHECKERBOARD & GRID) ===

        // Checkerboard pattern 1
        const checker1 = new Quad();
        checker1.bounds = new Bounds(new Point(370, 610), new Size(65, 80));
        checker1.background = Background.Pattern(
            0,
            Hsla.rgb(0.2, 0.2, 0.2, 1),
            Hsla.rgb(0.9, 0.9, 0.9, 1),
            20,
            2 // Checkerboard pattern
        );
        checker1.cornerRadii = Corners.uniform(8);
        checker1.contentMask.bounds = fullMask;
        scene.insertQuad(checker1);

        // Checkerboard pattern 2: colored
        const checker2 = new Quad();
        checker2.bounds = new Bounds(new Point(445, 610), new Size(65, 80));
        checker2.background = Background.Pattern(
            0,
            Hsla.rgb(0.8, 0.3, 0.3, 1),
            Hsla.rgb(0.3, 0.3, 0.8, 1),
            15,
            2 // Checkerboard pattern
        );
        checker2.cornerRadii = Corners.uniform(8);
        checker2.borderWidths = Edges.uniform(1);
        checker2.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
        checker2.contentMask.bounds = fullMask;
        scene.insertQuad(checker2);

        // Grid pattern 1
        const grid1 = new Quad();
        grid1.bounds = new Bounds(new Point(520, 610), new Size(65, 80));
        grid1.background = Background.Pattern(
            0,
            Hsla.rgb(0.95, 0.95, 0.98, 1),
            Hsla.rgb(0.3, 0.4, 0.6, 1),
            15,
            3 // Grid pattern
        );
        grid1.cornerRadii = Corners.uniform(8);
        grid1.contentMask.bounds = fullMask;
        scene.insertQuad(grid1);

        // === ROW 9: CONIC GRADIENTS ===

        // Conic gradient 1: Centered, starting at 0 degrees
        const conic1 = new Quad();
        conic1.bounds = new Bounds(new Point(370, 710), new Size(65, 80));
        conic1.background = Background.ConicGradient(
            0.5, 0.5, 0, // center, start angle 0
            [
                { color: Hsla.rgb(1.0, 0.3, 0.3, 1), position: 0 },
                { color: Hsla.rgb(0.3, 0.3, 1.0, 1), position: 1 }
            ],
            0 // sRGB
        );
        conic1.cornerRadii = Corners.uniform(32);
        conic1.contentMask.bounds = fullMask;
        scene.insertQuad(conic1);

        // Conic gradient 2: Offset center, Oklab
        const conic2 = new Quad();
        conic2.bounds = new Bounds(new Point(445, 710), new Size(65, 80));
        conic2.background = Background.ConicGradient(
            0.5, 0.5, 90, // center, start angle 90 degrees
            [
                { color: Hsla.rgb(0.3, 0.9, 0.3, 1), position: 0 },
                { color: Hsla.rgb(0.9, 0.3, 0.9, 1), position: 1 }
            ],
            1 // Oklab
        );
        conic2.cornerRadii = Corners.uniform(32);
        conic2.borderWidths = Edges.uniform(1);
        conic2.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
        conic2.contentMask.bounds = fullMask;
        scene.insertQuad(conic2);

        // Conic gradient 3: Multiple rotations with offset start
        const conic3 = new Quad();
        conic3.bounds = new Bounds(new Point(520, 710), new Size(65, 80));
        conic3.background = Background.ConicGradient(
            0.5, 0.5, 45, // center, start angle 45 degrees
            [
                { color: Hsla.rgb(0.9, 0.7, 0.2, 1), position: 0 },
                { color: Hsla.rgb(0.2, 0.7, 0.9, 1), position: 1 }
            ],
            1 // Oklab
        );
        conic3.cornerRadii = Corners.uniform(32);
        conic3.contentMask.bounds = fullMask;
        scene.insertQuad(conic3);

        // === ROW 10: OPACITY DEMONSTRATIONS ===

        // Opacity 1: Solid with 75% opacity
        const opacity1 = new Quad();
        opacity1.bounds = new Bounds(new Point(370, 810), new Size(65, 80));
        opacity1.background = Background.Solid(Hsla.rgb(0.9, 0.3, 0.3, 1));
        opacity1.cornerRadii = Corners.uniform(10);
        opacity1.opacity = 0.75;
        opacity1.contentMask.bounds = fullMask;
        scene.insertQuad(opacity1);

        // Opacity 2: Gradient with 50% opacity
        const opacity2 = new Quad();
        opacity2.bounds = new Bounds(new Point(445, 810), new Size(65, 80));
        opacity2.background = Background.LinearGradient(
            90,
            [
                { color: Hsla.rgb(0.3, 0.9, 0.3, 1), position: 0 },
                { color: Hsla.rgb(0.3, 0.3, 0.9, 1), position: 1 }
            ],
            1 // Oklab
        );
        opacity2.cornerRadii = Corners.uniform(10);
        opacity2.borderWidths = Edges.uniform(2);
        opacity2.borderColor = Hsla.rgb(0.2, 0.2, 0.2, 1);
        opacity2.opacity = 0.5;
        opacity2.contentMask.bounds = fullMask;
        scene.insertQuad(opacity2);

        // Opacity 3: Pattern with 30% opacity
        const opacity3 = new Quad();
        opacity3.bounds = new Bounds(new Point(520, 810), new Size(65, 80));
        opacity3.background = Background.Pattern(
            0,
            Hsla.rgb(0.2, 0.2, 0.2, 1),
            Hsla.rgb(0.9, 0.9, 0.9, 1),
            15,
            2 // Checkerboard
        );
        opacity3.cornerRadii = Corners.uniform(10);
        opacity3.opacity = 0.3;
        opacity3.contentMask.bounds = fullMask;
        scene.insertQuad(opacity3);

        // === PROCEDURAL IMAGES ===

        // Create procedural image sprites (these are async)
        const proceduralSprites = await Promise.all([
            imageLoader.createProceduralSprite(
                64, 64,
                ProceduralImages.gradientCircle,
                370, 455,
                { cornerRadii: Corners.uniform(8), opacity: 1.0 }
            ),
            imageLoader.createProceduralSprite(
                64, 64,
                ProceduralImages.geometric,
                445, 455,
                { cornerRadii: Corners.uniform(8), opacity: 1.0 }
            ),
            imageLoader.createProceduralSprite(
                64, 64,
                ProceduralImages.mandelbrot,
                520, 455,
                { cornerRadii: Corners.uniform(8), opacity: 1.0 }
            )
        ]);

        proceduralSprites.forEach(sprite => {
            sprite.contentMask.bounds = fullMask;
            scene.insertPolychromeSprite(sprite);
        });

        // === HIERARCHICAL CLIPPING DEMO ===

        // Create a parent container with clipping
        const clipContainer = new Quad();
        clipContainer.bounds = new Bounds(new Point(20, 710), new Size(310, 130));
        clipContainer.background = Background.Solid(Hsla.rgb(0.95, 0.95, 0.98, 1));
        clipContainer.cornerRadii = Corners.uniform(10);
        clipContainer.borderWidths = Edges.uniform(2);
        clipContainer.borderColor = Hsla.rgb(0.6, 0.6, 0.7, 1);
        clipContainer.contentMask.bounds = fullMask;
        scene.insertQuad(clipContainer);

        // Push clip region for the container
        scene.pushClip(new Bounds(new Point(25, 715), new Size(300, 120)));

        // Add children that will be clipped
        // This quad extends beyond the container bounds but will be clipped
        const clippedQuad1 = new Quad();
        clippedQuad1.bounds = new Bounds(new Point(15, 710), new Size(150, 135));
        clippedQuad1.background = Background.LinearGradient(90, [
            { color: Hsla.rgb(0.9, 0.3, 0.4, 1), position: 0 },
            { color: Hsla.rgb(0.4, 0.2, 0.8, 1), position: 1 }
        ], 1);
        clippedQuad1.cornerRadii = Corners.uniform(8);
        scene.insertQuad(clippedQuad1);

        // Second clipped quad
        const clippedQuad2 = new Quad();
        clippedQuad2.bounds = new Bounds(new Point(215, 710), new Size(125, 135));
        clippedQuad2.background = Background.LinearGradient(180, [
            { color: Hsla.rgb(0.3, 0.8, 0.5, 1), position: 0 },
            { color: Hsla.rgb(0.2, 0.4, 0.9, 1), position: 1 }
        ], 1);
        clippedQuad2.cornerRadii = Corners.uniform(8);
        scene.insertQuad(clippedQuad2);

        // Nested clip region (intersection with parent)
        scene.pushClip(new Bounds(new Point(50, 640), new Size(250, 70)));

        // Content in nested clip region
        const nestedQuad1 = new Quad();
        nestedQuad1.bounds = new Bounds(new Point(45, 635), new Size(110, 80));
        nestedQuad1.background = Background.Solid(Hsla.rgb(0.9, 0.8, 0.2, 1));
        nestedQuad1.cornerRadii = Corners.uniform(5);
        scene.insertQuad(nestedQuad1);

        const nestedQuad2 = new Quad();
        nestedQuad2.bounds = new Bounds(new Point(200, 635), new Size(110, 80));
        nestedQuad2.background = Background.Solid(Hsla.rgb(0.2, 0.9, 0.8, 1));
        nestedQuad2.cornerRadii = Corners.uniform(5);
        scene.insertQuad(nestedQuad2);

        // Add some sprites in the nested clip region
        const nestedSprite = new MonochromeSprite();
        nestedSprite.bounds = new Bounds(new Point(145, 660), new Size(32, 32));
        nestedSprite.color = Hsla.rgb(1.0, 0.5, 0.3, 1);
        nestedSprite.tile = starTile;
        scene.insertMonochromeSprite(nestedSprite);

        // Pop nested clip
        scene.popClip();

        // Add more content in parent clip (but not nested)
        const parentClippedQuad = new Quad();
        parentClippedQuad.bounds = new Bounds(new Point(250, 700), new Size(70, 50));
        parentClippedQuad.background = Background.Pattern(45,
            Hsla.rgb(0.9, 0.4, 0.5, 1),
            Hsla.rgb(1.0, 0.95, 0.95, 1),
            6,
            0);
        parentClippedQuad.cornerRadii = Corners.uniform(5);
        scene.insertQuad(parentClippedQuad);

        // Pop parent clip
        scene.popClip();

        // === TEXT LAYOUT DEMOS ===

        // Create text measurer
        const textMeasurer = new TextMeasurer();

        // Text wrapping demo
        const wrapContainer = new Quad();
        wrapContainer.bounds = new Bounds(new Point(340, 610), new Size(240, 130));
        wrapContainer.background = Background.Solid(Hsla.rgb(0.98, 0.97, 0.95, 1));
        wrapContainer.cornerRadii = Corners.uniform(8);
        wrapContainer.borderWidths = Edges.uniform(1);
        wrapContainer.borderColor = Hsla.rgb(0.7, 0.7, 0.7, 1);
        wrapContainer.contentMask.bounds = fullMask;
        scene.insertQuad(wrapContainer);

        // Wrapped text
        const wrappedText = 'Text layout utilities provide wrapping, alignment, and truncation for complex UI rendering.';
        const wrappedLines = textMeasurer.wrapText(wrappedText, 220, 11, 'sans-serif', 'normal');

        let lineY = 620;
        for (const line of wrappedLines) {
            const lineSprites = textRenderer.renderText(
                line,
                350, lineY,
                Hsla.rgb(0.2, 0.2, 0.2, 1),
                11,
                'sans-serif',
                'normal'
            );
            lineSprites.forEach(sprite => {
                sprite.contentMask.bounds = fullMask;
                scene.insertMonochromeSprite(sprite);
            });
            lineY += 16;
        }

        // Text alignment demo - centered text
        const centerContainer = new Quad();
        centerContainer.bounds = new Bounds(new Point(340, 750), new Size(120, 50));
        centerContainer.background = Background.Solid(Hsla.rgb(0.95, 0.95, 0.98, 1));
        centerContainer.cornerRadii = Corners.uniform(6);
        centerContainer.borderWidths = Edges.uniform(1);
        centerContainer.borderColor = Hsla.rgb(0.6, 0.6, 0.7, 1);
        centerContainer.contentMask.bounds = fullMask;
        scene.insertQuad(centerContainer);

        const centeredText = 'Centered';
        const centeredMeasurement = textMeasurer.measureText(centeredText, 12, 'sans-serif', 'bold');
        const centeredPos = getAlignedTextPosition(
            centeredMeasurement.width,
            centeredMeasurement.height,
            340, 750, 120, 50,
            TextAlign.Center,
            VerticalAlign.Middle
        );

        const centeredSprites = textRenderer.renderText(
            centeredText,
            centeredPos.x, centeredPos.y,
            Hsla.rgb(0.3, 0.4, 0.7, 1),
            12,
            'sans-serif',
            'bold'
        );
        centeredSprites.forEach(sprite => {
            sprite.contentMask.bounds = fullMask;
            scene.insertMonochromeSprite(sprite);
        });

        // Text truncation demo
        const truncContainer = new Quad();
        truncContainer.bounds = new Bounds(new Point(470, 750), new Size(110, 50));
        truncContainer.background = Background.Solid(Hsla.rgb(0.98, 0.95, 0.95, 1));
        truncContainer.cornerRadii = Corners.uniform(6);
        truncContainer.borderWidths = Edges.uniform(1);
        truncContainer.borderColor = Hsla.rgb(0.7, 0.6, 0.6, 1);
        truncContainer.contentMask.bounds = fullMask;
        scene.insertQuad(truncContainer);

        const longText = 'This text is too long to fit';
        const truncated = truncateText(longText, 100, 11, 'sans-serif', 'normal');
        const truncSprites = textRenderer.renderText(
            truncated,
            475, 770,
            Hsla.rgb(0.2, 0.2, 0.2, 1),
            11,
            'sans-serif',
            'normal'
        );
        truncSprites.forEach(sprite => {
            sprite.contentMask.bounds = fullMask;
            scene.insertMonochromeSprite(sprite);
        });

        // === DASHED BORDER DEMOS ===

        // Dashed border quad 1
        const dashedQuad1 = new Quad();
        dashedQuad1.bounds = new Bounds(new Point(370, 150), new Size(100, 80));
        dashedQuad1.background = Background.Solid(Hsla.rgb(0.98, 0.98, 1.0, 1));
        dashedQuad1.cornerRadii = Corners.uniform(10);
        dashedQuad1.borderWidths = Edges.uniform(2);
        dashedQuad1.borderColor = Hsla.rgb(0.3, 0.5, 0.9, 1);
        dashedQuad1.borderStyle = 1; // Dashed
        dashedQuad1.contentMask.bounds = fullMask;
        scene.insertQuad(dashedQuad1);

        // Dashed border quad 2 with varying widths
        const dashedQuad2 = new Quad();
        dashedQuad2.bounds = new Bounds(new Point(480, 150), new Size(100, 80));
        dashedQuad2.background = Background.LinearGradient(135, [
            { color: Hsla.rgb(1.0, 0.95, 0.9, 1), position: 0 },
            { color: Hsla.rgb(0.9, 0.85, 0.95, 1), position: 1 }
        ], 0);
        dashedQuad2.cornerRadii = Corners.uniform(12);
        dashedQuad2.borderWidths = new Edges(3, 3, 3, 3);
        dashedQuad2.borderColor = Hsla.rgb(0.9, 0.4, 0.5, 1);
        dashedQuad2.borderStyle = 1; // Dashed
        dashedQuad2.contentMask.bounds = fullMask;
        scene.insertQuad(dashedQuad2);

        // Dashed border card
        const dashedCard = new Quad();
        dashedCard.bounds = new Bounds(new Point(370, 240), new Size(210, 50));
        dashedCard.background = Background.Solid(Hsla.rgb(0.95, 0.98, 0.95, 1));
        dashedCard.cornerRadii = Corners.uniform(8);
        dashedCard.borderWidths = Edges.uniform(2);
        dashedCard.borderColor = Hsla.rgb(0.4, 0.7, 0.5, 1);
        dashedCard.borderStyle = 1; // Dashed
        dashedCard.contentMask.bounds = fullMask;
        scene.insertQuad(dashedCard);

        // === PATH RENDERING DEMOS ===

        // Path 1: Filled circle
        const circlePath = new Path(
            new PathBuilder().circle(60, 330, 20).build()
        );
        circlePath.fillColor = Hsla.rgb(0.3, 0.6, 0.9, 1);
        circlePath.filled = true;
        circlePath.stroked = false;
        circlePath.opacity = 1.0;
        circlePath.contentMask.bounds = fullMask;
        scene.insertPath(circlePath);

        // Path 2: Stroked rectangle with rounded corners
        const rectPath = new Path(
            new PathBuilder().roundedRect(110, 310, 60, 40, 8).build()
        );
        rectPath.strokeColor = Hsla.rgb(0.9, 0.4, 0.3, 1);
        rectPath.strokeWidth = 3;
        rectPath.filled = false;
        rectPath.stroked = true;
        rectPath.opacity = 1.0;
        rectPath.contentMask.bounds = fullMask;
        scene.insertPath(rectPath);

        // Path 3: Bezier curve (heart shape)
        const heartPath = new Path(
            new PathBuilder().moveTo(240, 330)
                .cubicTo(240, 315, 225, 310, 215, 310)
                .cubicTo(200, 310, 190, 320, 190, 330)
                .cubicTo(190, 340, 200, 350, 240, 370)
                .cubicTo(280, 350, 290, 340, 290, 330)
                .cubicTo(290, 320, 280, 310, 265, 310)
                .cubicTo(255, 310, 240, 315, 240, 330)
                .close()
                .build()
        );
        heartPath.fillColor = Hsla.rgb(0.95, 0.3, 0.5, 1);
        heartPath.filled = true;
        heartPath.stroked = false;
        heartPath.opacity = 1.0;
        heartPath.contentMask.bounds = fullMask;
        scene.insertPath(heartPath);

        // Path 4: SVG path (star)
        const starSVGPath = new Path(
            parseSVGPath('M 50 15 L 60 35 L 82 35 L 65 48 L 72 68 L 50 55 L 28 68 L 35 48 L 18 35 L 40 35 Z').build()
        );
        starSVGPath.fillColor = Hsla.rgb(0.9, 0.7, 0.2, 1);
        starSVGPath.strokeColor = Hsla.rgb(0.6, 0.4, 0.1, 1);
        starSVGPath.strokeWidth = 2;
        starSVGPath.filled = true;
        starSVGPath.stroked = true;
        starSVGPath.transform = Transform.translation(270, 295);
        starSVGPath.opacity = 1.0;
        starSVGPath.contentMask.bounds = fullMask;
        scene.insertPath(starSVGPath);

        // === TRANSFORM DEMOS ===

        // Rotating square
        const rotatingSquare = new Quad();
        rotatingSquare.bounds = new Bounds(new Point(420, 525), new Size(50, 50));
        rotatingSquare.background = Background.LinearGradient(90, [
            { color: Hsla.rgb(0.9, 0.3, 0.5, 1), position: 0 },
            { color: Hsla.rgb(0.5, 0.2, 0.8, 1), position: 1 }
        ], 1);
        rotatingSquare.cornerRadii = Corners.uniform(5);
        rotatingSquare.borderWidths = Edges.uniform(2);
        rotatingSquare.borderColor = Hsla.rgb(0.3, 0.1, 0.4, 1);
        rotatingSquare.contentMask.bounds = fullMask;
        scene.insertQuad(rotatingSquare);

        // Scaling pulse square
        const pulsingSquare = new Quad();
        pulsingSquare.bounds = new Bounds(new Point(485, 525), new Size(40, 40));
        pulsingSquare.background = Background.Solid(Hsla.rgb(0.3, 0.7, 0.9, 1));
        pulsingSquare.cornerRadii = Corners.uniform(8);
        pulsingSquare.contentMask.bounds = fullMask;
        scene.insertQuad(pulsingSquare);

        // Combined rotate + scale
        const combinedSquare = new Quad();
        combinedSquare.bounds = new Bounds(new Point(540, 525), new Size(35, 35));
        combinedSquare.background = Background.Solid(Hsla.rgb(0.9, 0.7, 0.2, 1));
        combinedSquare.cornerRadii = Corners.uniform(3);
        combinedSquare.contentMask.bounds = fullMask;
        scene.insertQuad(combinedSquare);

        // === TEXT LABELS ===

        // Title
        const titleSprites = textRenderer.renderText(
            'WebGPU Renderer',
            370, 50,
            Hsla.rgb(0.2, 0.3, 0.5, 1),
            24,
            'sans-serif',
            'bold'
        );
        titleSprites.forEach(sprite => {
            sprite.contentMask.bounds = fullMask;
            scene.insertMonochromeSprite(sprite);
        });

        // Subtitle
        const subtitleSprites = textRenderer.renderText(
            'GPU-accelerated UI primitives',
            370, 75,
            Hsla.rgb(0.4, 0.4, 0.4, 1),
            14,
            'sans-serif',
            'normal'
        );
        subtitleSprites.forEach(sprite => {
            sprite.contentMask.bounds = fullMask;
            scene.insertMonochromeSprite(sprite);
        });

        // Row labels
        const labels = [
            { text: 'sRGB Gradient', x: 45, y: 135 },
            { text: 'Oklab Gradient', x: 205, y: 135 },
            { text: 'Glyphs', x: 140, y: 210 },
            { text: 'Opacity', x: 265, y: 225, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Grayscale', x: 350, y: 225, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Faded', x: 450, y: 225, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Straight', x: 55, y: 297 },
            { text: 'Wavy', x: 240, y: 297 },
            { text: 'Composite', x: 145, y: 460 },
            { text: '45° stripes', x: 40, y: 560 },
            { text: '0° stripes', x: 160, y: 560 },
            { text: '135° stripes', x: 270, y: 560 },
            { text: 'Dots', x: 180, y: 600 },
            { text: 'Dashed Borders', x: 430, y: 145, size: 10, weight: 'bold', color: Hsla.rgb(0.4, 0.5, 0.8, 1) },
            { text: 'Vector Paths', x: 145, y: 305, size: 10, weight: 'bold', color: Hsla.rgb(0.6, 0.4, 0.7, 1) },
            { text: 'Procedural Images', x: 420, y: 450, size: 10, color: Hsla.rgb(0.4, 0.4, 0.4, 0.9) },
            { text: 'Hierarchical Clipping', x: 100, y: 750, size: 11, weight: 'bold', color: Hsla.rgb(0.5, 0.3, 0.7, 1) },
            { text: 'Text Layout', x: 410, y: 605, size: 10, weight: 'bold', color: Hsla.rgb(0.6, 0.5, 0.4, 1) },
            { text: 'Wrapping', x: 390, y: 740, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Align', x: 375, y: 805, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Truncate', x: 490, y: 805, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Transforms', x: 460, y: 520, size: 10, color: Hsla.rgb(0.4, 0.4, 0.4, 0.9) },
            { text: 'Rotate', x: 428, y: 580, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Scale', x: 490, y: 580, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Both', x: 545, y: 580, size: 8, color: Hsla.rgb(0.5, 0.5, 0.5, 0.8) },
            { text: 'Click any primitive to highlight it!', x: 20, y: 605, size: 11, weight: 'bold', color: Hsla.rgb(0.9, 0.5, 0.1, 1) }
        ];

        labels.forEach(label => {
            const sprites = textRenderer.renderText(
                label.text,
                label.x,
                label.y,
                label.color || Hsla.rgb(0.5, 0.5, 0.5, 0.8),
                label.size || 10,
                'sans-serif',
                label.weight || 'normal'
            );
            sprites.forEach(sprite => {
                sprite.contentMask.bounds = fullMask;
                scene.insertMonochromeSprite(sprite);
            });
        });

        // Initial render
        renderer.render(scene);

        let frame = 0;
        let mouseX = 0;
        let mouseY = 0;
        let mouseDown = false;
        let lastFrameTime = performance.now();
        let fps = 60;
        let fpsSprites = [];
        let trailQuads = [];
        const maxTrailLength = 20;

        // Bouncing balls physics
        const balls = [];
        const ballCount = 8;
        const ballArea = {
            x: 370,
            y: 150,
            width: 210,
            height: 290
        };

        // Create bouncing balls
        for (let i = 0; i < ballCount; i++) {
            const radius = 8 + Math.random() * 8;
            const ball = {
                x: ballArea.x + radius + Math.random() * (ballArea.width - radius * 2),
                y: ballArea.y + radius + Math.random() * (ballArea.height - radius * 2),
                vx: (Math.random() - 0.5) * 4,
                vy: (Math.random() - 0.5) * 4,
                radius: radius,
                color: Hsla.rgb(
                    Math.random() * 0.5 + 0.3,
                    Math.random() * 0.5 + 0.3,
                    Math.random() * 0.5 + 0.5,
                    0.9
                ),
                quad: new Quad(),
                shadow: new Shadow()
            };

            // Setup ball quad
            ball.quad.bounds = new Bounds(
                new Point(ball.x - ball.radius, ball.y - ball.radius),
                new Size(ball.radius * 2, ball.radius * 2)
            );
            ball.quad.background = Background.Solid(ball.color);
            ball.quad.cornerRadii = Corners.uniform(ball.radius);
            ball.quad.contentMask.bounds = fullMask;
            scene.insertQuad(ball.quad);

            // Setup ball shadow
            ball.shadow.bounds = new Bounds(
                new Point(ball.x - ball.radius + 2, ball.y - ball.radius + 2),
                new Size(ball.radius * 2, ball.radius * 2)
            );
            ball.shadow.cornerRadii = Corners.uniform(ball.radius);
            ball.shadow.blurRadius = 4;
            ball.shadow.color = Hsla.black(0.3);
            ball.shadow.contentMask.bounds = fullMask;
            scene.insertShadow(ball.shadow);

            balls.push(ball);
        }

        // Ball container border
        const ballContainer = new Quad();
        ballContainer.bounds = new Bounds(
            new Point(ballArea.x, ballArea.y),
            new Size(ballArea.width, ballArea.height)
        );
        ballContainer.background = Background.Solid(Hsla.rgb(0.95, 0.95, 0.98, 1));
        ballContainer.borderWidths = Edges.uniform(2);
        ballContainer.borderColor = Hsla.rgb(0.6, 0.6, 0.7, 1);
        ballContainer.cornerRadii = Corners.uniform(10);
        ballContainer.contentMask.bounds = fullMask;
        scene.insertQuad(ballContainer);

        // Move ball container behind balls
        scene.quads.splice(scene.quads.indexOf(ballContainer), 1);
        scene.quads.splice(0, 0, ballContainer);

        info.textContent = 'Rendering complete!';

        // Create hit tester
        const hitTester = new HitTester(scene);

        // Highlight quad for showing clicked primitives
        let highlightQuad = null;
        let highlightedPrimitive = null;

        // Mouse tracking
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
        });

        canvas.addEventListener('mousedown', () => {
            mouseDown = true;
        });

        canvas.addEventListener('mouseup', () => {
            mouseDown = false;
        });

        // Click handler for hit testing
        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const hitResult = hitTester.hitTest(x, y);

            // Remove previous highlight
            if (highlightQuad) {
                const index = scene.quads.indexOf(highlightQuad);
                if (index > -1) {
                    scene.quads.splice(index, 1);
                }
                highlightQuad = null;
            }

            if (hitResult) {
                console.log('Hit:', hitResult.type, 'at index', hitResult.index);

                // Create highlight based on primitive type
                if (hitResult.type === 'quad') {
                    const quad = hitResult.primitive;
                    highlightQuad = new Quad();
                    highlightQuad.bounds = new Bounds(
                        new Point(quad.bounds.origin.x - 4, quad.bounds.origin.y - 4),
                        new Size(quad.bounds.size.width + 8, quad.bounds.size.height + 8)
                    );
                    highlightQuad.background = Background.Solid(Hsla.rgb(1.0, 0.8, 0.0, 0));
                    highlightQuad.cornerRadii = quad.cornerRadii;
                    highlightQuad.borderWidths = Edges.uniform(3);
                    highlightQuad.borderColor = Hsla.rgb(1.0, 0.8, 0.0, 1);
                    highlightQuad.contentMask.bounds = fullMask;
                    highlightQuad.transform = quad.transform;
                    scene.insertQuad(highlightQuad);
                    highlightedPrimitive = quad;
                } else if (hitResult.type === 'monochromeSprite' || hitResult.type === 'polychromeSprite') {
                    const sprite = hitResult.primitive;
                    highlightQuad = new Quad();
                    highlightQuad.bounds = new Bounds(
                        new Point(sprite.bounds.origin.x - 3, sprite.bounds.origin.y - 3),
                        new Size(sprite.bounds.size.width + 6, sprite.bounds.size.height + 6)
                    );
                    highlightQuad.background = Background.Solid(Hsla.rgb(0.0, 1.0, 1.0, 0));
                    highlightQuad.cornerRadii = Corners.uniform(8);
                    highlightQuad.borderWidths = Edges.uniform(2);
                    highlightQuad.borderColor = Hsla.rgb(0.0, 1.0, 1.0, 1);
                    highlightQuad.contentMask.bounds = fullMask;
                    scene.insertQuad(highlightQuad);
                    highlightedPrimitive = sprite;
                } else if (hitResult.type === 'underline') {
                    const underline = hitResult.primitive;
                    highlightQuad = new Quad();
                    highlightQuad.bounds = new Bounds(
                        new Point(underline.bounds.origin.x - 2, underline.bounds.origin.y - 2),
                        new Size(underline.bounds.size.width + 4, underline.bounds.size.height + 4)
                    );
                    highlightQuad.background = Background.Solid(Hsla.rgb(1.0, 0.0, 1.0, 0));
                    highlightQuad.cornerRadii = Corners.uniform(4);
                    highlightQuad.borderWidths = Edges.uniform(2);
                    highlightQuad.borderColor = Hsla.rgb(1.0, 0.0, 1.0, 1);
                    highlightQuad.contentMask.bounds = fullMask;
                    scene.insertQuad(highlightQuad);
                    highlightedPrimitive = underline;
                } else if (hitResult.type === 'shadow') {
                    const shadow = hitResult.primitive;
                    highlightQuad = new Quad();
                    highlightQuad.bounds = new Bounds(
                        new Point(shadow.bounds.origin.x - 2, shadow.bounds.origin.y - 2),
                        new Size(shadow.bounds.size.width + 4, shadow.bounds.size.height + 4)
                    );
                    highlightQuad.background = Background.Solid(Hsla.rgb(0.5, 0.5, 0.5, 0));
                    highlightQuad.cornerRadii = shadow.cornerRadii;
                    highlightQuad.borderWidths = Edges.uniform(2);
                    highlightQuad.borderColor = Hsla.rgb(0.5, 0.5, 0.5, 1);
                    highlightQuad.contentMask.bounds = fullMask;
                    scene.insertQuad(highlightQuad);
                    highlightedPrimitive = shadow;
                }
            } else {
                console.log('No hit');
                highlightedPrimitive = null;
            }
        });

        // Mouse cursor indicator
        const cursorQuad = new Quad();
        cursorQuad.bounds = new Bounds(new Point(0, 0), new Size(8, 8));
        cursorQuad.background = Background.Solid(Hsla.rgb(1.0, 0.3, 0.3, 0.8));
        cursorQuad.cornerRadii = Corners.uniform(4);
        cursorQuad.contentMask.bounds = fullMask;
        scene.insertQuad(cursorQuad);

        // Animation loop
        const animate = () => {
            frame++;

            // Calculate FPS
            const currentTime = performance.now();
            const deltaTime = currentTime - lastFrameTime;
            lastFrameTime = currentTime;
            fps = Math.round(1000 / deltaTime);

            // Remove old FPS sprites from scene
            fpsSprites.forEach(sprite => {
                const index = scene.monochromeSprites.indexOf(sprite);
                if (index > -1) {
                    scene.monochromeSprites.splice(index, 1);
                }
            });

            // Render new FPS counter
            fpsSprites = textRenderer.renderText(
                `FPS: ${fps}`,
                370, 110,
                Hsla.rgb(0.3, 0.6, 0.3, 1),
                14,
                'monospace',
                'bold'
            );
            fpsSprites.forEach(sprite => {
                sprite.contentMask.bounds = fullMask;
                scene.insertMonochromeSprite(sprite);
            });

            // Animate gradients
            gradQuad1.background.gradientAngle = 45 + Math.sin(frame * 0.02) * 45;
            gradQuad2.background.gradientAngle = 90 + Math.sin(frame * 0.03) * 30;

            // Animate corner radii
            const radius1 = 15 + Math.sin(frame * 0.04) * 8;
            gradQuad1.cornerRadii = Corners.uniform(radius1);

            // Animate shadows
            shadow1.blurRadius = 6 + Math.sin(frame * 0.05) * 3;
            shadow2.blurRadius = 8 + Math.sin(frame * 0.04) * 4;

            // Animate borders
            const borderPulse = 3 + Math.sin(frame * 0.06) * 2;
            gradQuad1.borderWidths = Edges.uniform(borderPulse);

            // Animate sprite positions
            const bounce = Math.sin(frame * 0.05) * 10;
            circleSprite.bounds.origin.y = 150 + bounce;
            starSprite.bounds.origin.y = 150 + Math.sin(frame * 0.05 + 1) * 10;
            heartSprite.bounds.origin.y = 150 + Math.sin(frame * 0.05 + 2) * 10;

            // Animate sprite colors
            const hue = (frame * 0.01) % 1;
            circleSprite.color = Hsla.rgb(
                0.9 * Math.abs(Math.sin(hue * Math.PI * 2)),
                0.2,
                0.3,
                1
            );

            // Animate polychrome sprite opacity
            gradientSprite.opacity = 0.6 + Math.sin(frame * 0.03) * 0.4;

            // Animate faded sprite opacity
            fadedSprite.opacity = 0.3 + Math.sin(frame * 0.04) * 0.3;

            // Toggle grayscale periodically
            if (Math.floor(frame / 60) % 4 < 2) {
                grayscaleSprite.grayscale = true;
            } else {
                grayscaleSprite.grayscale = false;
            }

            // Animate underline thickness
            wavyUnderline1.thickness = 4 + Math.sin(frame * 0.04) * 2;

            // Animate pattern backgrounds
            pattern1.background.gradientAngle = 45 + Math.sin(frame * 0.02) * 15;
            pattern2.background.colors[0].percentage = 8 + Math.sin(frame * 0.03) * 4;

            // Animate dot patterns
            dotPattern1.background.colors[0].percentage = 12 + Math.sin(frame * 0.025) * 3;
            dotPattern2.background.colors[0].percentage = 10 + Math.sin(frame * 0.03) * 2;

            // Animate transforms
            // Rotation transform (rotate around center)
            const rotAngle = frame * 0.02;
            const rotCenter = new Point(
                rotatingSquare.bounds.origin.x + rotatingSquare.bounds.size.width / 2,
                rotatingSquare.bounds.origin.y + rotatingSquare.bounds.size.height / 2
            );
            rotatingSquare.transform = Transform.translation(rotCenter.x, rotCenter.y)
                .multiply(Transform.rotation(rotAngle))
                .multiply(Transform.translation(-rotCenter.x, -rotCenter.y));

            // Scale transform (pulse)
            const scaleValue = 1.0 + Math.sin(frame * 0.04) * 0.3;
            const scaleCenter = new Point(
                pulsingSquare.bounds.origin.x + pulsingSquare.bounds.size.width / 2,
                pulsingSquare.bounds.origin.y + pulsingSquare.bounds.size.height / 2
            );
            pulsingSquare.transform = Transform.translation(scaleCenter.x, scaleCenter.y)
                .multiply(Transform.scale(scaleValue))
                .multiply(Transform.translation(-scaleCenter.x, -scaleCenter.y));

            // Combined rotate + scale
            const combAngle = frame * 0.03;
            const combScale = 1.0 + Math.sin(frame * 0.05) * 0.4;
            const combCenter = new Point(
                combinedSquare.bounds.origin.x + combinedSquare.bounds.size.width / 2,
                combinedSquare.bounds.origin.y + combinedSquare.bounds.size.height / 2
            );
            combinedSquare.transform = Transform.translation(combCenter.x, combCenter.y)
                .multiply(Transform.rotation(combAngle))
                .multiply(Transform.scale(combScale))
                .multiply(Transform.translation(-combCenter.x, -combCenter.y));

            // Update cursor position
            cursorQuad.bounds.origin.x = mouseX - 4;
            cursorQuad.bounds.origin.y = mouseY - 4;

            // Change cursor appearance on mouse down
            if (mouseDown) {
                cursorQuad.bounds.size = new Size(12, 12);
                cursorQuad.cornerRadii = Corners.uniform(6);
                cursorQuad.background = Background.Solid(Hsla.rgb(0.3, 0.7, 1.0, 0.9));

                // Create trail effect when mouse is down
                if (frame % 2 === 0) {
                    const trailQuad = new Quad();
                    const size = 4 + Math.random() * 4;
                    trailQuad.bounds = new Bounds(
                        new Point(mouseX - size / 2, mouseY - size / 2),
                        new Size(size, size)
                    );
                    trailQuad.background = Background.Solid(
                        Hsla.rgb(
                            0.3 + Math.random() * 0.4,
                            0.5 + Math.random() * 0.3,
                            0.8 + Math.random() * 0.2,
                            0.6
                        )
                    );
                    trailQuad.cornerRadii = Corners.uniform(size / 2);
                    trailQuad.contentMask.bounds = fullMask;

                    trailQuads.push({
                        quad: trailQuad,
                        life: 1.0
                    });

                    scene.insertQuad(trailQuad);
                }
            } else {
                cursorQuad.bounds.size = new Size(8, 8);
                cursorQuad.cornerRadii = Corners.uniform(4);
                cursorQuad.background = Background.Solid(Hsla.rgb(1.0, 0.3, 0.3, 0.8));
            }

            // Update trail particles
            for (let i = trailQuads.length - 1; i >= 0; i--) {
                const trail = trailQuads[i];
                trail.life -= 0.05;

                // Update alpha
                const color = trail.quad.background.solid;
                color.a = trail.life * 0.6;

                // Update size (shrink)
                const newSize = trail.quad.bounds.size.width * 0.95;
                trail.quad.bounds.size = new Size(newSize, newSize);
                trail.quad.cornerRadii = Corners.uniform(newSize / 2);

                // Remove dead particles
                if (trail.life <= 0) {
                    const index = scene.quads.indexOf(trail.quad);
                    if (index > -1) {
                        scene.quads.splice(index, 1);
                    }
                    trailQuads.splice(i, 1);
                }
            }

            // Limit trail length
            while (trailQuads.length > maxTrailLength) {
                const oldTrail = trailQuads.shift();
                const index = scene.quads.indexOf(oldTrail.quad);
                if (index > -1) {
                    scene.quads.splice(index, 1);
                }
            }

            // Update bouncing balls physics
            const gravity = 0.2;
            const damping = 0.99;
            const restitution = 0.8;

            for (let i = 0; i < balls.length; i++) {
                const ball = balls[i];

                // Apply gravity
                ball.vy += gravity;

                // Apply damping
                ball.vx *= damping;
                ball.vy *= damping;

                // Update position
                ball.x += ball.vx;
                ball.y += ball.vy;

                // Wall collisions
                if (ball.x - ball.radius < ballArea.x) {
                    ball.x = ballArea.x + ball.radius;
                    ball.vx = Math.abs(ball.vx) * restitution;
                }
                if (ball.x + ball.radius > ballArea.x + ballArea.width) {
                    ball.x = ballArea.x + ballArea.width - ball.radius;
                    ball.vx = -Math.abs(ball.vx) * restitution;
                }
                if (ball.y - ball.radius < ballArea.y) {
                    ball.y = ballArea.y + ball.radius;
                    ball.vy = Math.abs(ball.vy) * restitution;
                }
                if (ball.y + ball.radius > ballArea.y + ballArea.height) {
                    ball.y = ballArea.y + ballArea.height - ball.radius;
                    ball.vy = -Math.abs(ball.vy) * restitution;
                }

                // Ball-to-ball collisions
                for (let j = i + 1; j < balls.length; j++) {
                    const other = balls[j];
                    const dx = other.x - ball.x;
                    const dy = other.y - ball.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const minDist = ball.radius + other.radius;

                    if (dist < minDist) {
                        // Normalize collision vector
                        const nx = dx / dist;
                        const ny = dy / dist;

                        // Separate balls
                        const overlap = minDist - dist;
                        ball.x -= nx * overlap * 0.5;
                        ball.y -= ny * overlap * 0.5;
                        other.x += nx * overlap * 0.5;
                        other.y += ny * overlap * 0.5;

                        // Relative velocity
                        const dvx = other.vx - ball.vx;
                        const dvy = other.vy - ball.vy;
                        const dvn = dvx * nx + dvy * ny;

                        // Don't resolve if velocities are separating
                        if (dvn < 0) {
                            // Collision impulse
                            const impulse = (2 * dvn) / 2;
                            ball.vx += impulse * nx * restitution;
                            ball.vy += impulse * ny * restitution;
                            other.vx -= impulse * nx * restitution;
                            other.vy -= impulse * ny * restitution;
                        }
                    }
                }

                // Mouse interaction - push balls away
                if (mouseDown) {
                    const dx = ball.x - mouseX;
                    const dy = ball.y - mouseY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const pushRadius = 50;

                    if (dist < pushRadius && dist > 0) {
                        const force = (1 - dist / pushRadius) * 2;
                        ball.vx += (dx / dist) * force;
                        ball.vy += (dy / dist) * force;
                    }
                }

                // Update quad position
                ball.quad.bounds.origin.x = ball.x - ball.radius;
                ball.quad.bounds.origin.y = ball.y - ball.radius;

                // Update shadow position (slightly offset)
                ball.shadow.bounds.origin.x = ball.x - ball.radius + 2;
                ball.shadow.bounds.origin.y = ball.y - ball.radius + 2;
            }

            // Re-render
            renderer.render(scene);

            const primitiveCount =
                scene.shadows.length +
                scene.quads.length +
                scene.underlines.length +
                scene.monochromeSprites.length +
                scene.polychromeSprites.length +
                scene.paths.length;

            const bufferStats = renderer.bufferPool.getStats();
            info.textContent = `Frame ${frame} | Primitives: ${primitiveCount} | Buffers: ${bufferStats.total} (${bufferStats.inUse} in use, ${bufferStats.available} available, ${bufferStats.totalSizeMB}MB) | FPS: ${fps}`;

            requestAnimationFrame(animate);
        };

        // Start animation
        requestAnimationFrame(animate);

    } catch (error) {
        console.error('Error:', error);
        errorDiv.textContent = `Error: ${error.message}\n\n${error.stack}\n\nMake sure you're using a browser with WebGPU support (Chrome 113+).`;
        errorDiv.style.display = 'block';
        info.textContent = 'Failed to initialize';
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
} else {
    main();
}
