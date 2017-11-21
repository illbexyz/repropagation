type matrix('a) = array(array('a));

let rows = (matrix) => Array.length(matrix);

let cols = (matrix) => Array.length(matrix[0]);

let mult = (x, n) => {
  let x_rows = rows(x);
  let x_cols = cols(x);
  let z = Array.make_matrix(x_rows, x_cols, 0.0);
  for (i in 0 to x_rows - 1) {
    for (j in 0 to x_cols - 1) {
      z[i][j] = x[i][j] *. n
    }
  };
  z
};

let sum = (x, y) => {
  let x_rows = rows(x);
  let x_cols = cols(x);
  let z = Array.make_matrix(x_rows, x_cols, 0.0);
  for (i in 0 to x_rows - 1) {
    for (j in 0 to x_cols - 1) {
      z[i][j] = x[i][j] +. y[i][j]
    }
  };
  z
};

let dot = (x, y) => {
  let x_rows = Array.length(x);
  let y_rows = Array.length(y);
  let y_cols = max(0, Array.length(y[0]));
  let z = Array.make_matrix(x_rows, y_cols, 0.0);
  for (i in 0 to x_rows - 1) {
    for (j in 0 to y_cols - 1) {
      for (k in 0 to y_rows - 1) {
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