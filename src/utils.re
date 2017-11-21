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

let diff_loss = (expected, output) => {
    (output -. expected)
};