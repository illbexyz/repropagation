let dot = (x, y) => {
    let x0 = Array.length(x)
    and y0 = Array.length(y);
    let y1 =
      if (y0 == 0) {
        0
      } else {
        Array.length(y[0])
      };
    let z = Array.make_matrix(x0, y1, 0.0);
    for (i in 0 to x0 - 1) {
      for (j in 0 to y1 - 1) {
        for (k in 0 to y0 - 1) {
          z[i][j] = z[i][j] +. x[i][k] *. y[k][j]
        }
      }
    };
    z
  };

let apply = (matrix, fn) => {
  Array.map(
    (i) => Array.map((j) => fn(j), i),
    matrix
  )
};