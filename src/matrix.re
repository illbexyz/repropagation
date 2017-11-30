type matrix('a) = array(array('a));

let rows = (matrix) => Array.length(matrix);

let cols = (matrix) => Array.length(matrix[0]);

let mat_op = (x, y, fn) => {
  let x_rows = rows(x);
  let x_cols = cols(x);
  let z = Array.make_matrix(x_rows, x_cols, 0.0);
  for (i in 0 to x_rows - 1) {
    for (j in 0 to x_cols - 1) {
      z[i][j] = fn(x[i][j], y[i][j])
    }
  };
  z
};

let sum = (x, y) => {
  mat_op(x, y, (a, b) => a +. b)
};

let sub = (x, y) => {
  mat_op(x, y, (a, b) => a -. b)
};

let mult = (x, y) => {
  if (cols(x) != rows(y)) {
    raise(invalid_arg("the x rows and y columns must match"))
  };
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

let transpose = (matrix) => {
  Array.mapi(
    (iCol, _) => Array.map((row) => row[iCol], matrix),
    matrix[0]
  )
};
