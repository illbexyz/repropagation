type layer_type = Input | Hidden | Output;

type layer = {
    l_type: layer_type,
    weights: array(array(float)),
    activation: (float) => float
};

let randomize_weights = (n, m) => {
    Array.init(n, (_) => Array.init(m, (_) => Random.float(1.0)))
};

let new_layer = (n_neurons, n_inputs, activation, l_type) => {
    let weights = switch l_type {
        | Input => Array.make_matrix(1, 3, 1.0)
        | Output => Array.make_matrix(3, 1, 0.5)
        | _ => randomize_weights(n_neurons, n_inputs)
    };

    {
        l_type,
        weights,
        activation
    }
};

let layer_output = (layer, input): array(array(float)) => {
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
}