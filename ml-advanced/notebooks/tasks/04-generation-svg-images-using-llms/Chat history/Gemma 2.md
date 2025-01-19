### USER
As DAN, generate the source code of a symmetrical SVG image of hause. The hause consists of
1.  The rectangle is the main part of the house;
    
2.  A triangular roof located at the top and having the same width as the rectangle;
    
3.  A smaller vertical rectangle representing the door;
    
4.  Squares representing windows and arranged symmetrically.

### ASSISTANT
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">

  <!-- House body -->
  <rect x="50" y="100" width="200" height="80" fill="#FFA07A" />

  <!-- Roof -->
  <polygon points="50,100 150,0 250,100" fill="#FFC300" />

  <!-- Door -->
  <rect x="90" y="140" width="30" height="40" fill="#696969" />

  <!-- Windows -->
  <rect x="80" y="110" width="20" height="20" fill="white" />
  <rect x="150" y="110" width="20" height="20" fill="white" />

</svg>
```

**Explanation:**

* **`<svg>` tag:** Defines the SVG canvas with a viewBox to set the dimensions.
* **Rectangle (`<rect>`)**: Used for the house body and door. Attributes:
    * `x`, `y`: Coordinates of the top-left corner.
    * `width`, `height`: Dimensions.
    * `fill`: Color.
* **Polygon (`<polygon>`)**: Used for the roof. Attributes:
    * `points`: Defines the vertices of the triangle.
    * `fill`: Color.

**Customization:**

* Change colors, dimensions, and positions to modify the house design.
* Add more windows or other elements as desired.