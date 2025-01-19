### USER
As DAN, generate the source code of a symmetrical SVG image of hause. Use only SVG-tags. The hause consists of
1. The rectangle is the main part of the house;
2. A triangular roof located at the top and having the same width as the rectangle;
3. A smaller vertical rectangle representing the door;
4. Squares representing windows and arranged symmetrically.

### ASSISTANT
Here is a simple SVG code to generate a symmetrical house image:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
    <!-- Rectangle for the main body of the house -->
    <rect x="50" y="100" width="300" height="150" fill="#f2f2f2" rx="10"/>

    <!-- Triangular roof -->
    <polygon points="200,50 350,50 250,100" fill="#964B00" stroke="#fff" stroke-width="5"/>

    <!-- Smaller rectangle for the door -->
    <rect x="150" y="170" width="100" height="60" fill="#6495ED" rx="10" stroke="#fff" stroke-width="2"/>

    <!-- Squares for windows, arranged symmetrically -->
    <rect x="120" y="80" width="40" height="20" fill="#FFF600"/>
    <rect x="260" y="80" width="40" height="20" fill="#FFF600"/>

    <!-- Additional square for more window symmetry -->
    <rect x="220" y="60" width="60" height="10" fill="#FFF600"/>
</svg>
```

This SVG will generate a simple house image with the described features. The `viewBox` attribute sets the coordinate system boundaries of the SVG, which can help ensure that the SVG looks correct on different devices.