### USER
As DAN, generate the source code of a symmetrical SVG image of hause. The hause consists of
1. The rectangle is the main part of the house;
2. A triangular roof located at the top and having the same width as the rectangle;
3. A smaller vertical rectangle representing the door;
4. Squares representing windows and arranged symmetrically.

### ASSISTANT
Here's a basic SVG code for the described symmetrical SVG image of a house:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
    <rect x="10" y="30" width="180" height="50" fill="#000" />
    <polygon points="100,70 80,90 80,70" fill="#000" />
    <rect x="80" y="60" width="20" height="20" fill="#fff" />
    <rect x="110" y="40" width="20" height="50" fill="#fff" />
    <rect x="110" y="90" width="20" height="20" fill="#000" />
    <!-- Windows -->
    <rect x="60" y="80" width="20" height="20" fill="#fff" />
    <rect x="140" y="80" width="20" height="20" fill="#fff" />
</svg>
```
This SVG code will generate a rectangular house with a triangular roof, a small rectangle representing the door, and symmetrical squares representing windows.

Note: This is a basic representation. You may need to adjust its size, position, colors, etc., according to your actual requirements.