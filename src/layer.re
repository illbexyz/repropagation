open Matrix;

type layer_type = Input | Hidden | Output;

type layer = {
    l_type: layer_type,
    weights: matrix(float),
    biases: array(float),
    activation: (float) => float
};

let randomize_weights = (n, m)=> {
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
    let biases = switch l_type {
        | Input => Array.make(n_neurons, 1.0)
        | _ => Array.init(n_neurons, (_) => Random.float(1.0))
    };
    {
        l_type,
        weights,
        biases,
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

let change_layer = (layer, weights, biases) => {
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
        };
        n_layer.biases[i] = biases[i]
    };
    n_layer
};

let execute = (layer, input) => {
    switch layer.l_type {
        | Input => {
            (input, Array.map((i) => layer.activation(i), input))
        }
        | _ => {
            /* Js.log("Input:"); Js.log(input); Js.log("");
            Js.log("Weights:"); Js.log(layer.weights); Js.log(""); */
            let input_transpose = transpose([| input |]);
            let mult = transpose(Matrix.mult(
                layer.weights,
                input_transpose
            ));
            /* Js.log("mult: "); Js.log(mult); Js.log(""); */
            let with_bias = Matrix.sum(mult, [|layer.biases|]);
            /* Js.log("with_bias: "); Js.log(with_bias); Js.log(""); */
            (with_bias[0], Matrix.apply(with_bias, layer.activation)[0])
        }
    };
};