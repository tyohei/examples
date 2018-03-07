const data = [
  {"x": 1, "y": 2},
  {"x": 5, "y": 7},
  {"x": 3, "y": 3},
  {"x": 9, "y": 2},
  {"x": 4, "y": 3},
  {"x": 2, "y": 3},
  {"x": 5, "y": 5},
  {"x": 3, "y": 4},
  {"x": 4, "y": 3},
  {"x": 7, "y": 4},
  {"x": 6, "y": -2}
];
data.sort(function(a, b) {
  if (a["x"] > b["x"]) return 1;
  else return -1;
});
const dataIterator = data.entries();

let xMin;
let xMax;
let yMin;
let yMax;
for (let [i, pair] of dataIterator) {
  if (i == 0) {
    xMin = pair["x"];
    xMax = pair["x"];
    yMin = pair["y"];
    yMax = pair["y"];
  } else {
    if (xMin > pair["x"]) xMin = pair["x"];
    if (xMax < pair["x"]) xMax = pair["x"];
    if (yMin > pair["y"]) yMin = pair["y"];
    if (yMax < pair["y"]) yMax = pair["y"];
  }
}

document.writeln("yMin: " + yMin);
document.writeln("yMax: " + yMax);


const svg_width  = 1000;
const svg_height = 350;
const margin     = {top: 20, bottom: 20, left: 30, right: 20};
const width      = svg_width - margin.left - margin.right;
const height     = svg_height - margin.top - margin.bottom;

let svg = d3.select("body").append("svg")
          .attr("width", svg_width)
          .attr("height", svg_height)
          .append("g")

svg.attr("transform", "translate(" + margin.left + ", " + margin.right + ")");
document.writeln(typeof(svg))

let xScale = d3.scaleLinear()
             .domain([xMin, xMax])
             .range([0, width]);
document.writeln(typeof(xScale))

let yScale = d3.scaleLinear()
             .domain([yMin, yMax])
             .range([height, 0]);
document.writeln(typeof(yScale))

svg.append("g")  // Create <g> tag
   .call(
     d3.axisLeft(yScale).ticks(data.length)
   )
   .append("text")
   .text("y axis")
   .attr("stroke", "#000")
   .attr("x", -50)
   .attr("y", 0);

svg.append("g")
  .attr("transform", "translate(0, " + height + ")")
  .call(
    d3.axisBottom(xScale).ticks(data.length)
  )
  .append("text")
  .text("x axis")
  .attr("x", width)
  .attr("y", margin.bottom - 10);

let line = d3.line()
           .x(function(d, i) { return xScale(d.x); })
           .y(function(d, i) { return yScale(d.y); });
// (d, i) -> (data, index)

svg.append("path")
   .datum(data)
   .attr("d", line)
   .attr("fill","none")
   .attr("stroke","#000");
