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

let quadratic_difference = (~expected, ~actual) => {
    (actual -. expected) ** 2.0
};

let array_map2 = (fn, x, y) => {
    if (Array.length(x) != Array.length(y)) {
        raise(invalid_arg("x and y have not the same length"))
    };
    Array.mapi((i, _) => fn(x[i], y[i]), x)
};

let std_deviation = (expecteds, actuals) => {
    let quad_diffs = array_map2((a, e) => quadratic_difference(~expected=e, ~actual=a), actuals, expecteds);
    let sum = Array.fold_left((a, b) => a +. b, 0.0, quad_diffs);
    sqrt(sum /. float_of_int(Array.length(actuals)))
};

let vectorize1 = (fn) => {
    (x) => Array.map((xi) => fn(xi), x)
};

let vectorize2 = (fn) => {
    (x, y) => Array.map((xi) => fn(xi, y), x)
};

let vectorize2_both = (fn) => {
    (x, y) => array_map2((a, b) => fn(a, b), x, y)
};

let adamard = (x, y) => {
    array_map2((a, b) => a *. b, x, y)
};

let array_average = (x) => {
    let sum = Array.fold_left((prev, curr) => prev +. curr, 0.0, x);
    sum /. float_of_int(Array.length(x))
};