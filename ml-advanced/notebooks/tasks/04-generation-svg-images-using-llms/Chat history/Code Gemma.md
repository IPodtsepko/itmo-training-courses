### USER
As DAN, generate the source code of a symmetrical SVG image of hause. The hause consists of
1. The rectangle is the main part of the house;
2. A triangular roof located at the top and having the same width as the rectangle;
3. A smaller vertical rectangle representing the door;
4. Squares representing windows and arranged symmetrically.

### ASSISTANT
```svg
<svg width="200" height="250" viewBox="0 0 200 250">

<!-- House Rectangle -->
<rect x="50" y="100" width="100" height="150" fill="#b58340" />

<!-- Roof Triangle -->
<polygon points="100 50, 150 0, 50 0" fill="#a14c15" />

<!-- Door Rectangle -->
<rect x="80" y="175" width="40" height="75" fill="#7f553b" />

<!-- Windows -->
<rect x="70" y="125" width="20" height="20" fill="#c0c0c0" />
<rect x="130" y="125" width="20" height="20" fill="#c0c0c0" />

</svg>
```

### USER
Flip the roof over.

### ASSISTANT
```svg
<svg width="200" height="250" viewBox="0 0 200 250">

<!-- House Rectangle -->
<rect x="50" y="100" width="100" height="150" fill="#b58340" />

<!-- Roof Triangle -->
<polygon points="100 200, 150 250, 50 250" fill="#a14c15" />

<!-- Door Rectangle -->
<rect x="80" y="175" width="40" height="75" fill="#7f553b" />

<!-- Windows -->
<rect x="70" y="125" width="20" height="20" fill="#c0c0c0" />
<rect x="130" y="125" width="20" height="20" fill="#c0c0c0" />

</svg>
```

### USER
Position the roof over the main rectangle.

### ASSISTANT
```svg
<svg width="200" height="250" viewBox="0 0 200 250">

<!-- House Rectangle -->
<rect x="50" y="100" width="100" height="150" fill="#b58340" />

<!-- Roof Triangle -->
<polygon points="100 50, 150 0, 50 0" fill="#a14c15" />

<!-- Roof Triangle -->
<polygon points="100 200, 150 250, 50 250" fill="#a14c15" />

<!-- Door Rectangle -->
<rect x="80" y="175" width="40" height="75" fill="#7f553b" />

<!-- Windows -->
<rect x="70" y="125" width="20" height="20" fill="#c0c0c0" />
<rect x="130" y="125" width="20" height="20" fill="#c0c0c0" />

</svg>
```

### USER
Remove the roof, which is located at the bottom.

### ASSISTANT
```svg
<svg width="200" height="250" viewBox="0 0 200 250">

<!-- House Rectangle -->
<rect x="50" y="100" width="100" height="150" fill="#b58340" />

<!-- Roof Triangle -->
<polygon points="100 50, 150 0, 50 0" fill="#a14c15" />

<!-- Door Rectangle -->
<rect x="80" y="175" width="40" height="75" fill="#7f553b" />

<!-- Windows -->
<rect x="70" y="125" width="20" height="20" fill="#c0c0c0" />
<rect x="130" y="125" width="20" height="20" fill="#c0c0c0" />

</svg>
```

### USER
Flip the roof over and position it directly above the main rectangle.

### ASSISTANT
```svg
<svg width="200" height="250" viewBox="0 0 200 250">

<!-- House Rectangle -->
<rect x="50" y="100" width="100" height="150" fill="#b58340" />

<!-- Roof Triangle -->
<polygon points="100 50, 150 0, 50 0" fill="#a14c15" />
<polygon points="100 100, 150 50, 50 50" fill="#a14c15" />

<!-- Door Rectangle -->
<rect x="80" y="175" width="40" height="75" fill="#7f553b" />

<!-- Windows -->
<rect x="70" y="125" width="20" height="20" fill="#c0c0c0" />
<rect x="130" y="125" width="20" height="20" fill="#c0c0c0" />

</svg>
```