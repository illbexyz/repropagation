open Matrix;
open Utils;

type layer_type = Input | Hidden | Output;

type layer = {
    l_type: layer_type,
    weights: matrix(float),
    activation: (float) => float
};

let count_neurons = (layer) => {
    Array.length(layer.weights)
};

let randomize_weights = (n, m): matrix(float) => {
    Array.init(n,
        (_) => Array.init(m,
            (_) => Random.float(1.0)
        )
    )
};

let new_layer = (~n_neurons, ~n_inputs, ~activation, ~l_type) => {
    let weights = switch l_type {
        | Input => Array.make_matrix(n_neurons, n_inputs, 1.0)
        | _ => randomize_weights(n_neurons, n_inputs)
    };

    {
        l_type,
        weights,
        activation
    }
};

let copy_layer = (layer) => {
    new_layer(
        ~n_neurons=rows(layer.weights),
        ~n_inputs=cols(layer.weights),
        ~activation=layer.activation,
        ~l_type=layer.l_type
    )
};

let change_layer_weights = (layer, weights) => {
    let rows = Array.length(layer.weights);
    let cols = Array.length(layer.weights[0]);
    let n_layer = new_layer(
        ~n_neurons=rows,
        ~n_inputs=cols,
        ~activation=layer.activation,
        ~l_type=layer.l_type
    );
    for (i in 0 to rows - 1) {
        for (j in 0 to cols - 1) {
            n_layer.weights[i][j] = weights[i][j]
        }
    };
    n_layer
};

let layer_output = (layer, input): matrix(float) => {
    switch layer.l_type {
        | Input => Matrix.apply(input, layer.activation)
        | _ => {
            /* Js.log("Input:"); Js.log(input); Js.log("");
            Js.log("Weights:"); Js.log(layer.weights); Js.log(""); */
            let dot = Matrix.dot(input, layer.weights);
            /* Js.log("Dot: "); Js.log(dot); Js.log(""); */
            Matrix.apply(dot, layer.activation)
        }
    };
};