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

const margin = {
  top: 30,
  left: 30,
};

let svg = d3.select("body").append("svg")
          .attr("width", 1000)
          .attr("height", 350)
