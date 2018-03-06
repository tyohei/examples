var svg = d3.select("body")        // Select <body>...</body> tag as "svg"
          .append("svg")           // creates <svg>...</svg>
          .attr("width", 500)      // add width attribute to <svg>
          .attr("height", 300);    // add height attribute to <svg>


/* svg is something like a canvas to draw. */
var circles = svg.append("circle")  // creates <circle>...</circle>
             .attr('cx', 100)      // center x
             .attr('cy', 100)      // center y
             .attr('r', 10)        // rectangle
             .attr('fill', '#f0f');  // fill
