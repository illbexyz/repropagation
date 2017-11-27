let e = 2.7182818284590452353602874713527;

let identity = (x) => x;

let relu = (x) => {
    x > 0.0 ?
        x
        :
        0.0
};

let diff_relu = (x) => {
    x > 0.0 ?
        1.0
        :
        0.0
};

let logistic = (x) => {
      1.0 /. (1.0 +. e ** (x))
};

let diff_logistic = (x) => {
    logistic(x) *. (1.0 -. logistic(x))
};

let diff_loss = (expected, actual) => {
    (actual -. expected)
};

let quadratic_difference = (expected, actual) => {
    (actual -. expected) ** 2.0
};

let std_deviation = (~expecteds, ~actuals) => {
    let quad_diffs = Array.mapi((i, actual) => quadratic_difference(expecteds[i], actual) , actuals);
    let sum = Array.fold_left((a, b) => a +. b, 0.0, quad_diffs);
    sqrt(sum /. float_of_int(Array.length(actuals)))
};

let vec_dot = (x, y) => {
    Array.mapi((i, _) => x[i] *. y[i], x)
};

let invert_vec = (x) => {
    Array.map((v) => 1.0 -. v, x)
};

let vectorize1 = (fn) => {
    (x) => Array.map((xi) => fn(xi), x)
};

let vectorize2 = (fn) => {
    (x, y) => Array.map((xi) => fn(xi, y), x)
};

let adamard = (x, y) => {
    Array.mapi((i, xi) => xi *. y[i], x)
};

let sub (x, y) = {
    x - y
};